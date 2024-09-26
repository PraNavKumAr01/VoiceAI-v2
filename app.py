from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from graph import WorkFlow
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from TTS import text_to_speech
import base64
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.environ.get("MONGO_URL")

client = MongoClient(uri, server_api=ServerApi('1'))
db = client[os.environ.get("DB_NAME")]
conversations = db[os.environ.get("DB_FOLDER")]

class QueryRequest(BaseModel):
    question: str
    user_id: str
    conversation_id: str
    image: Optional[str]

class ChatHistoryRequest(BaseModel):
    user_id: str
    conversation_id: str
    
app = FastAPI()

origins = [
    "http://localhost:3000/session",
    "http://localhost:3000",
    "https://vox-ai-bay.vercel.app",
    "https://voxai-beta.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = WorkFlow().local_agent

def format_chat_history(chat_history):
    formatted_history = ""
    for pair in chat_history:
        formatted_history += f"Human: {pair[0]}\nAI: {pair[1]}\n"
    return formatted_history

@app.post("/get-chat-history")
async def get_chat_history(request: ChatHistoryRequest):
    try:
        existing_records = list(conversations.find({"user_id": request.user_id, "conversation_id": request.conversation_id}).sort("timestamp"))
        if existing_records:
            chat_history = [{"Question": record["user_message"], "Response": record["agent_response"]} for record in existing_records if record["user_message"] and record["agent_response"]]
        else:
            chat_history = "Chat history not found"

        return {
            "chat_history" : chat_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/text-to-llm")
async def text_to_llm(request: QueryRequest):
    try:
        existing_records = list(conversations.find({"user_id": request.user_id, "conversation_id": request.conversation_id}).sort("timestamp"))

        if existing_records:
            records = [[record["user_message"], record["agent_response"]] for record in existing_records if record["user_message"] and record["agent_response"]]
            chat_history = format_chat_history(records)
        else:
            chat_history = ""
        
        if request.image:
            image_data = request.image
            image_is = True
        else:
            image_data = ""
            image_is = False

        result = agent.invoke({"question" : request.question,"imageAvailable": image_is, "image" : image_data ,"chat_history" : chat_history})

        audio_bytes = text_to_speech(result["generation"])

        ai_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        record = {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "timestamp": datetime.now(),
            "user_message": request.question,
            "agent_response": result["generation"]
        }

        conversations.insert_one(record)

        return {
            "answer" : result["generation"],
            "audio_bytes": ai_audio_base64,
            "links" : result["video_links"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
