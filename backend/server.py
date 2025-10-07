from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import uuid
import json
from datetime import datetime
from pathlib import Path
# from azure.storage.blob import BlobServiceClient, ContentSettings
import boto3
from botocore.exceptions import ClientError

from context import prompt

load_dotenv(override=True)

app = FastAPI()

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client = AzureOpenAI(
#     api_version=os.getenv("API_VERSION"),
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name=os.getenv("DEFAULT_AWS_REGION", "us-east-1")
)
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")


# USE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"
# AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "")

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")


# if USE_AZURE:
#         blob_service_client = BlobServiceClient.from_connection_string(os.getenv('BLOB_STORAGE_CONNECTION_STRING'))
#         container_client = blob_service_client.get_container_client(os.getenv('BLOB_CONTAINER_NAME'))

if USE_S3:
    s3_client = boto3.client("s3")

# def load_personality():
#     with open("me.txt", 'r', encoding='utf-8') as f:
#         return f.read().strip()

# PERSONALITY = load_personality()

class ChatRequest(BaseModel):
    message: str
    session_id : Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"

def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from file"""

    # if USE_AZURE:
    #     try:
    #         blob_client = container_client.get_blob_client(get_memory_path(session_id))
    #         return json.loads(blob_client.download_blob().readall().decode('utf-8'))

    #     except Exception as e:
    #         print(f"Blob not found: {e}")
    #         return []  
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        file_path = MEMORY_DIR / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return []

def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to file"""
    # if USE_AZURE:
    #     json_data = json.dumps(messages, indent=2)
    #     blob_client = container_client.get_blob_client(get_memory_path(session_id))
    #     blob_client.upload_blob(json_data, overwrite=True, content_settings=ContentSettings(content_type="application/json"))
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = MEMORY_DIR / f"{session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)


def call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Call AWS Bedrock with conversation history"""
    
    # Build messages in Bedrock format
    messages = []
    
    # Add system prompt as first user message (Bedrock convention)
    messages.append({
        "role": "user", 
        "content": [{"text": f"System: {prompt()}"}]
    })
    
    # Add conversation history (limit to last 10 exchanges to manage context)
    for msg in conversation[-20:]:  # Last 10 back-and-forth exchanges
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })
    
    try:
        # Call Bedrock using the converse API
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.7,
                "topP": 0.9
            }
        )

        # response = client.chat.completions.create(
        #     model=os.getenv("DEPLOYMENT"),
        #     messages=messages
        # )
        # return response.choices[0].message.content
        
        # Extract the response text
        return response["output"]["message"]["content"][0]["text"]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            # Handle message format issues
            print(f"Bedrock validation error: {e}")
            raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
        elif error_code == 'AccessDeniedException':
            print(f"Bedrock access denied: {e}")
            raise HTTPException(status_code=403, detail="Access denied to Bedrock model")
        else:
            print(f"Bedrock error: {e}")
            raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")


@app.get("/")
async def root():
    return {"message" : "AI Digital Twin API",
            "memory_enabled": True,
            "storage": "Blob" if USE_S3 else "local"
            }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "use_azure": USE_S3}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Load conversation history
        conversation = load_conversation(session_id)

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message)

        # Update conversation history
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)