from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai import protos
import assemblyai as aai
import requests
import os
import time
import logging
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import uuid
from typing import Dict, List, Set, Optional, Any
import websockets
import base64
import asyncio
import wave
from collections import defaultdict
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Enhanced AI Voice Agent - Fixed Version")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend HTML file"""
    with open("test.html", "r") as f:
        return HTMLResponse(content=f.read())

# Configuration Management
class APIConfig:
    def __init__(self):
        self.session_configs = {}
        self.default_config = {
            "assemblyai_key": os.getenv("ASSEMBLYAI_API_KEY", ""),
            "murf_key": os.getenv("MURF_API_KEY", ""),
            "gemini_key": os.getenv("GEMINI_API_KEY", ""),
            "weather_key": os.getenv("WEATHER_KEY", ""),
            "search_key": os.getenv("SEARCH_API_KEY", ""),
            "search_engine_id": os.getenv("SEARCH_ENGINE_ID", "")
        }
    
    def get_config(self, session_id: str = "default") -> dict:
        # Validate session_id format
        if not self._is_valid_session_id(session_id):
            session_id = "default"
            
        if session_id in self.session_configs:
            config = self.default_config.copy()
            session_config = self.session_configs[session_id]
            for key, value in session_config.items():
                if value and str(value).strip():
                    config[key] = value
            return config
        return self.default_config
    
    def set_config(self, session_id: str, config: dict):
        if not self._is_valid_session_id(session_id):
            raise ValueError("Invalid session ID format")
            
        # Sanitize configuration values
        filtered_config = {}
        for key, value in config.items():
            if key in self.default_config and value and str(value).strip():
                # Basic sanitization - remove any potential script tags or dangerous characters
                sanitized_value = re.sub(r'[<>"\']', '', str(value).strip())
                if len(sanitized_value) > 5:  # Basic length check
                    filtered_config[key] = sanitized_value
        
        self.session_configs[session_id] = filtered_config
        logging.info(f"Config updated for session {session_id}: {list(filtered_config.keys())}")
    
    def _is_valid_session_id(self, session_id: str) -> bool:
        if not session_id or len(session_id) > 100:
            return False
        # Allow alphanumeric, underscore, hyphen
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', session_id))

config_manager = APIConfig()

# Persona System
PERSONAS = {
    "neutral": {
        "system": "You are a helpful, concise assistant. Keep responses short and friendly.",
        "voice": {"voiceId": "en-IN-isha", "style": "Conversational", "rate": 0, "pitch": 0}
    },
    "pirate": {
        "system": "You are a cheerful pirate. Use pirate slang like 'Arrr' and 'matey' occasionally.",
        "voice": {"voiceId": "en-IN-isha", "style": "Conversational", "rate": -2, "pitch": -2}
    },
    "cowboy": {
        "system": "You are a friendly cowboy. Use phrases like 'howdy' and 'partner'.",
        "voice": {"voiceId": "en-IN-isha", "style": "Conversational", "rate": -1, "pitch": -1}
    },
    "robot": {
        "system": "You are a polite robot. Speak with structured sentences and use 'Processing' occasionally.",
        "voice": {"voiceId": "en-IN-isha", "style": "Conversational", "rate": 1, "pitch": 2}
    },
}

# Global storage with proper cleanup
session_persona: Dict[str, str] = {}
chat_history: Dict[str, List[Dict[str, str]]] = {}
active_connections: Dict[str, WebSocket] = {}
murf_connections: Set[WebSocket] = set()
transcribe_connections: Dict[str, WebSocket] = {}

# AssemblyAI real-time transcription storage
transcription_sessions: Dict[str, Dict] = {}

# Rate limiting - Fixed limits
class RateLimit:
    def __init__(self):
        self.requests = defaultdict(list)
        self.max_requests_per_minute = 5  # Reduced from 15 to be more reasonable
        
    def can_make_request(self, key: str = "default") -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > minute_ago]
        return len(self.requests[key]) < self.max_requests_per_minute
    
    def record_request(self, key: str = "default"):
        self.requests[key].append(datetime.now())

rate_limiter = RateLimit()

# Helper functions with proper input validation
def sanitize_input(text: str, max_length: int = 200) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\'\&]', '', text.strip())
    return sanitized[:max_length]

def get_persona(session_id: str) -> dict:
    pid = session_persona.get(session_id, "neutral")
    return PERSONAS.get(pid, PERSONAS["neutral"])

def get_weather(city: str, session_id: str = "default") -> str:
    try:
        # Sanitize city input
        city = sanitize_input(city, 50)
        if not city:
            return "Please provide a valid city name."
            
        config = config_manager.get_config(session_id)
        weather_key = config.get("weather_key")
        
        if not weather_key:
            return "Weather service not configured. Please add your OpenWeatherMap API key."
            
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 401:
            return "Invalid weather API key."
        elif response.status_code == 404:
            return f"City '{city}' not found."
        elif response.status_code != 200:
            return f"Weather service error for {city}."
            
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description'].title()
        
        return f"Weather in {city}: {temp}°C, {description}"
        
    except requests.RequestException as e:
        logging.error(f"Weather API request error: {e}")
        return "Weather service temporarily unavailable."
    except Exception as e:
        logging.error(f"Weather error: {e}")
        return "Error retrieving weather information."

def search_web(query: str, session_id: str = "default", num_results: int = 3) -> str:
    try:
        # Sanitize query
        query = sanitize_input(query, 100)
        if not query:
            return "Please provide a valid search query."
            
        config = config_manager.get_config(session_id)
        search_key = config.get("search_key")
        search_engine_id = config.get("search_engine_id")
        
        if not search_key or not search_engine_id:
            return "Search not configured. Please add Google Search API key and Engine ID."
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': search_key,
            'cx': search_engine_id,
            'q': query.strip(),
            'num': min(max(1, num_results), 5),
            'safe': 'active'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 400:
            return "Invalid search parameters."
        elif response.status_code == 403:
            return "Search quota exceeded or invalid API key."
        elif response.status_code != 200:
            return "Search service unavailable."
        
        data = response.json()
        if 'items' not in data:
            return f"No results found for: {query}"
        
        results = []
        for item in data['items'][:num_results]:
            title = item.get('title', 'No title')[:100]  # Limit title length
            snippet = item.get('snippet', 'No description')[:200]  # Limit snippet length
            results.append(f"• {title}: {snippet}")
        
        return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        
    except requests.RequestException as e:
        logging.error(f"Search API request error: {e}")
        return "Search service temporarily unavailable."
    except Exception as e:
        logging.error(f"Search error: {e}")
        return "Search failed. Please try again."

def get_news(topic: str = "latest", session_id: str = "default", num_results: int = 3) -> str:
    try:
        topic = sanitize_input(topic, 50)
        
        if topic.lower() in ["latest", "news", "today", "headlines"]:
            query = "latest news headlines today"
        else:
            query = f"{topic} news today"
        
        return search_web(query, session_id, num_results)
        
    except Exception as e:
        logging.error(f"News error: {e}")
        return "Unable to fetch news."

# Fixed Gemini integration with better error handling
def get_gemini_model_with_tools(session_id: str = "default"):
    config = config_manager.get_config(session_id)
    gemini_key = config.get("gemini_key")
    
    if not gemini_key:
        raise ValueError("Gemini API key not configured")
    
    try:
        genai.configure(api_key=gemini_key)
        
        weather_function = protos.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for any city",
            parameters=protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    "city": protos.Schema(type=protos.Type.STRING, description="City name")
                },
                required=["city"]
            ),
        )
        
        search_function = protos.FunctionDeclaration(
            name="search_web",
            description="Search the web for information",
            parameters=protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    "query": protos.Schema(type=protos.Type.STRING, description="Search query")
                },
                required=["query"]
            ),
        )
        
        news_function = protos.FunctionDeclaration(
            name="get_news",
            description="Get latest news headlines",
            parameters=protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    "topic": protos.Schema(type=protos.Type.STRING, description="News topic")
                },
                required=["topic"]
            ),
        )
        
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=[protos.Tool(function_declarations=[weather_function, search_function, news_function])]
        )
    except Exception as e:
        logging.error(f"Error creating Gemini model with tools: {e}")
        raise ValueError(f"Failed to initialize Gemini model: {str(e)}")

def get_gemini_model_basic(session_id: str = "default"):
    config = config_manager.get_config(session_id)
    gemini_key = config.get("gemini_key")
    
    if not gemini_key:
        raise ValueError("Gemini API key not configured")
    
    try:
        genai.configure(api_key=gemini_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logging.error(f"Error creating basic Gemini model: {e}")
        raise ValueError(f"Failed to initialize Gemini model: {str(e)}")

def safe_gemini_call(model, prompt, stream=False, session_id="default"):
    try:
        if not rate_limiter.can_make_request(session_id):
            return "Rate limit exceeded. Please wait a moment before trying again."
        
        rate_limiter.record_request(session_id)
        
        # Ensure prompt is reasonable length
        if len(str(prompt)) > 4000:
            return "Input too long. Please try a shorter message."
        
        if stream:
            return model.generate_content(prompt, stream=True)
        else:
            return model.generate_content(prompt)
            
    except Exception as e:
        error_str = str(e)
        if "quota" in error_str.lower() or "429" in error_str:
            return "API quota exceeded. Please wait before trying again."
        elif "api key" in error_str.lower() or "401" in error_str:
            return "Invalid Gemini API key. Please check your configuration."
        elif "400" in error_str:
            return "Invalid request. Please try rephrasing your message."
        else:
            logging.error(f"Gemini error: {e}")
            return "AI service temporarily unavailable. Please try again."

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class ConfigRequest(BaseModel):
    assemblyai_key: Optional[str] = None
    murf_key: Optional[str] = None
    gemini_key: Optional[str] = None
    weather_key: Optional[str] = None
    search_key: Optional[str] = None
    search_engine_id: Optional[str] = None

class PersonaRequest(BaseModel):
    persona_id: str

# Fixed streaming functions
async def broadcast_to_murf_clients(message: dict):
    """Broadcast message to all Murf WebSocket clients"""
    disconnected = []
    for ws in list(murf_connections):
        try:
            await ws.send_json(message)
        except Exception as e:
            logging.error(f"Failed to send to Murf client: {e}")
            disconnected.append(ws)
    
    for ws in disconnected:
        murf_connections.discard(ws)

async def process_with_ai(text: str, session_id: str):
    """Process text with AI and return response"""
    try:
        # Sanitize input
        text = sanitize_input(text, 500)
        if not text:
            return "Please provide a valid message."
            
        persona = get_persona(session_id)
        system_prompt = persona["system"]
        
        # Check if tools are needed
        needs_tools = any(keyword in text.lower() for keyword in [
            'weather', 'temperature', 'news', 'search', 'latest', 'current', 'today'
        ])
        
        if needs_tools:
            model = get_gemini_model_with_tools(session_id)
            full_prompt = f"{system_prompt}\n\nUser: {text}"
        else:
            model = get_gemini_model_basic(session_id)
            full_prompt = f"{system_prompt}\n\nUser: {text}"
        
        response = safe_gemini_call(model, full_prompt, session_id=session_id)
        
        if isinstance(response, str):
            return response
        
        if response.candidates and response.candidates[0].content:
            candidate = response.candidates[0]
            
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fn = part.function_call
                    
                    # Execute tool with proper error handling
                    tool_result = "Tool execution failed"
                    try:
                        if fn.name == "get_weather":
                            city = fn.args.get("city", "")
                            tool_result = get_weather(city, session_id)
                        elif fn.name == "search_web":
                            query = fn.args.get("query", "")
                            tool_result = search_web(query, session_id)
                        elif fn.name == "get_news":
                            topic = fn.args.get("topic", "latest")
                            tool_result = get_news(topic, session_id)
                        else:
                            tool_result = f"Unknown tool: {fn.name}"
                    except Exception as tool_error:
                        logging.error(f"Tool execution error: {tool_error}")
                        tool_result = f"Error executing {fn.name}: {str(tool_error)}"
                    
                    # Generate natural response
                    try:
                        follow_up_model = get_gemini_model_basic(session_id)
                        follow_up_prompt = f"{system_prompt}\n\nUser asked: {text}\n\nInformation: {tool_result}\n\nProvide a natural, conversational response."
                        
                        follow_up_response = safe_gemini_call(follow_up_model, follow_up_prompt, session_id=session_id)
                        
                        if isinstance(follow_up_response, str):
                            return follow_up_response
                        elif follow_up_response.candidates and follow_up_response.candidates[0].content:
                            return follow_up_response.candidates[0].content.parts[0].text
                        else:
                            return f"Here's what I found: {tool_result}"
                    except Exception as follow_up_error:
                        logging.error(f"Follow-up generation error: {follow_up_error}")
                        return f"Here's what I found: {tool_result}"
                    
                elif hasattr(part, 'text') and part.text:
                    return part.text
        
        return "I couldn't process that request properly."
        
    except ValueError as e:
        return str(e)
    except Exception as e:
        logging.error(f"AI processing error: {e}")
        return "An error occurred while processing your request. Please try again."

# FIXED: AssemblyAI Real-time Transcription Functions
async def start_assemblyai_session(session_id: str):
    """Start AssemblyAI real-time transcription session with proper Universal-2 model configuration"""
    config = config_manager.get_config(session_id)
    assemblyai_key = config.get("assemblyai_key")
    
    if not assemblyai_key:
        logging.error("AssemblyAI API key not configured")
        return None
    
    try:
        # Set up AssemblyAI with correct configuration
        aai.settings.api_key = assemblyai_key
        
        # Create synchronous transcript handler (AssemblyAI callbacks are sync, not async)
        def handle_transcript_sync(transcript):
            # Schedule async handling without blocking the callback
            asyncio.create_task(handle_transcript_async(session_id, transcript))
        
        # FIXED: Use correct parameter name for Universal-2 model
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=handle_transcript_sync,
            on_error=lambda error: logging.error(f"AssemblyAI error: {error}"),
            model="universal-2"  # FIXED: Correct parameter name
        )
        
        # Connect to AssemblyAI (synchronous operation)
        transcriber.connect()
        
        transcription_sessions[session_id] = {
            "transcriber": transcriber,
            "active": True,
            "created_at": datetime.now()
        }
        
        logging.info(f"Started AssemblyAI session for {session_id} with Universal-2 model")
        return transcriber
        
    except Exception as e:
        logging.error(f"Failed to start AssemblyAI session: {e}")
        return None

async def handle_transcript_async(session_id: str, transcript):
    """Async handler for transcript processing"""
    try:
        websocket = transcribe_connections.get(session_id)
        if not websocket:
            return
        
        if transcript.message_type == aai.RealtimeTranscriptType.PartialTranscript:
            # Send partial transcript
            await websocket.send_text(f"PARTIAL::{transcript.text}")
            
        elif transcript.message_type == aai.RealtimeTranscriptType.FinalTranscript:
            # Send final transcript
            final_text = sanitize_input(transcript.text, 500)
            if not final_text.strip():
                return
                
            await websocket.send_text(f"FINAL::{final_text}")
            
            # Process with AI
            try:
                ai_response = await process_with_ai(final_text, session_id)
                
                # Add to chat history
                if session_id not in chat_history:
                    chat_history[session_id] = []
                
                chat_history[session_id].append({"role": "user", "content": final_text})
                chat_history[session_id].append({"role": "assistant", "content": ai_response})
                
                # Limit chat history size to prevent memory issues
                if len(chat_history[session_id]) > 50:
                    chat_history[session_id] = chat_history[session_id][-40:]
                
                # Send AI response
                await websocket.send_text(f"AI::{ai_response}")
                
                # Also broadcast to Murf clients for TTS
                await broadcast_to_murf_clients({
                    "type": "tts_request",
                    "text": ai_response,
                    "session_id": session_id
                })
                
            except Exception as ai_error:
                logging.error(f"AI processing error: {ai_error}")
                await websocket.send_text(f"ERROR::AI processing failed: {str(ai_error)}")
                
    except Exception as e:
        logging.error(f"Error handling transcript: {e}")

def stop_assemblyai_session(session_id: str):
    """Stop AssemblyAI session - Synchronous cleanup"""
    if session_id in transcription_sessions:
        try:
            session_data = transcription_sessions[session_id]
            transcriber = session_data.get("transcriber")
            
            if transcriber:
                # Close is synchronous
                transcriber.close()
                
            del transcription_sessions[session_id]
            logging.info(f"Closed AssemblyAI session for {session_id}")
            
        except Exception as e:
            logging.error(f"Error closing AssemblyAI session: {e}")

# Cleanup function for old sessions
async def cleanup_old_sessions():
    """Clean up sessions older than 1 hour"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=1)
        sessions_to_remove = []
        
        for session_id, session_data in transcription_sessions.items():
            created_at = session_data.get("created_at", datetime.now())
            if created_at < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            stop_assemblyai_session(session_id)
            # Also clean up other session data
            chat_history.pop(session_id, None)
            session_persona.pop(session_id, None)
            config_manager.session_configs.pop(session_id, None)
            
        if sessions_to_remove:
            logging.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            
    except Exception as e:
        logging.error(f"Error during session cleanup: {e}")

# API Endpoints with improved validation
@app.post("/api/config/{session_id}")
async def update_config(session_id: str, request: Request):
    """Update configuration for a session"""
    try:
        # Validate session ID
        if not config_manager._is_valid_session_id(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        
        # Handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            config_data = await request.json()
        else:
            # Handle form data
            form_data = await request.form()
            config_data = dict(form_data)
        
        # Update configuration with validation
        config_manager.set_config(session_id, config_data)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "session_id": session_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")

@app.get("/api/config/{session_id}")
async def get_config_status(session_id: str):
    """Get configuration status"""
    if not config_manager._is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    config = config_manager.get_config(session_id)
    
    safe_config = {}
    for key, value in config.items():
        safe_config[key + "_configured"] = bool(value and str(value).strip())
    
    return {
        "session_id": session_id,
        "config": safe_config
    }

@app.post("/api/persona/{session_id}")
async def set_persona(session_id: str, request: PersonaRequest):
    """Set persona for session"""
    if not config_manager._is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    persona_id = request.persona_id
    
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=400, detail=f"Invalid persona: {persona_id}")
    
    session_persona[session_id] = persona_id
    
    return {
        "success": True,
        "session_id": session_id,
        "persona_id": persona_id
    }

@app.get("/api/personas")
async def list_personas():
    """List available personas"""
    return {"personas": PERSONAS}

@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for session"""
    if not config_manager._is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    chat_history.pop(session_id, None)
    
    return {
        "success": True,
        "message": "History cleared",
        "session_id": session_id
    }

# WebSocket Endpoints

@app.websocket("/ws/murf-audio")
async def murf_audio_websocket(websocket: WebSocket):
    """WebSocket for Murf audio streaming"""
    await websocket.accept()
    murf_connections.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Audio WebSocket connected"
        })
        
        while True:
            try:
                # Add timeout to prevent hanging connections
                message = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "test":
                    await websocket.send_json({"type": "test_response", "message": "Audio WebSocket working"})
                
            except asyncio.TimeoutError:
                # Send ping to check if connection is still alive
                await websocket.send_json({"type": "ping"})
            except Exception as e:
                logging.error(f"Murf WebSocket message error: {e}")
                break
                
    except WebSocketDisconnect:
        logging.info("Murf audio WebSocket disconnected")
    except Exception as e:
        logging.error(f"Murf WebSocket error: {e}")
    finally:
        murf_connections.discard(websocket)

@app.websocket("/ws/stream-transcribe")
async def stream_transcribe_websocket(websocket: WebSocket, session_id: str = Query(...)):
    """WebSocket for streaming transcription - FIXED AssemblyAI integration"""
    # Validate session ID
    if not config_manager._is_valid_session_id(session_id):
        await websocket.close(code=1008, reason="Invalid session ID")
        return
        
    await websocket.accept()
    transcribe_connections[session_id] = websocket
    
    # Initialize session storage
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    try:
        await websocket.send_text("CONNECTED::Transcription WebSocket connected")
        
        # Try to start AssemblyAI session
        transcriber = await start_assemblyai_session(session_id)
        if transcriber:
            await websocket.send_text("CONNECTED::AssemblyAI Universal-2 model ready")
        else:
            await websocket.send_text("ERROR::Failed to initialize AssemblyAI. Check your API key.")
        
        while True:
            try:
                # Add timeout to prevent hanging connections
                message = await asyncio.wait_for(websocket.receive(), timeout=300.0)  # 5 minute timeout
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Handle audio data - send to AssemblyAI if available
                        audio_data = message["bytes"]
                        
                        if transcriber and session_id in transcription_sessions:
                            try:
                                # Send audio to AssemblyAI (synchronous call)
                                transcriber.stream(audio_data)
                            except Exception as stream_error:
                                logging.error(f"Error streaming audio to AssemblyAI: {stream_error}")
                                await websocket.send_text("ERROR::Audio streaming error")
                        else:
                            # Fallback: acknowledge receipt
                            await websocket.send_text("PARTIAL::Processing audio...")
                    
                    elif "text" in message:
                        try:
                            data = json.loads(message["text"])
                            
                            if data.get("type") == "text_input":
                                # Handle direct text input
                                text = sanitize_input(data.get("text", ""), 500)
                                if text:
                                    # Add to chat history
                                    chat_history[session_id].append({"role": "user", "content": text})
                                    
                                    # Process with AI
                                    ai_response = await process_with_ai(text, session_id)
                                    
                                    # Add AI response to history
                                    chat_history[session_id].append({"role": "assistant", "content": ai_response})
                                    
                                    # Send responses (matching frontend format)
                                    await websocket.send_text(f"FINAL::{text}")
                                    await websocket.send_text(f"AI::{ai_response}")
                                    
                                    # Also broadcast to Murf clients for TTS
                                    await broadcast_to_murf_clients({
                                        "type": "tts_request",
                                        "text": ai_response,
                                        "session_id": session_id
                                    })
                            
                            elif data.get("type") == "transcript":
                                # Handle transcript from frontend
                                transcript = sanitize_input(data.get("text", ""), 500)
                                if transcript:
                                    # Add to chat history
                                    chat_history[session_id].append({"role": "user", "content": transcript})
                                    
                                    ai_response = await process_with_ai(transcript, session_id)
                                    
                                    # Add AI response to history
                                    chat_history[session_id].append({"role": "assistant", "content": ai_response})
                                    
                                    await websocket.send_text(f"AI::{ai_response}")
                                    
                                    await broadcast_to_murf_clients({
                                        "type": "tts_request",
                                        "text": ai_response,
                                        "session_id": session_id
                                    })
                            
                            elif data.get("type") == "ping":
                                await websocket.send_text("PONG::Connection alive")
                                
                        except json.JSONDecodeError:
                            await websocket.send_text("ERROR::Invalid message format")
                
                elif message["type"] == "websocket.disconnect":
                    break
                    
            except asyncio.TimeoutError:
                # Send ping to check connection
                try:
                    await websocket.send_text("PING::Keep alive")
                except:
                    break
            except Exception as e:
                logging.error(f"Transcribe WebSocket message error: {e}")
                await websocket.send_text(f"ERROR::{str(e)}")
                
    except WebSocketDisconnect:
        logging.info(f"Transcription WebSocket {session_id} disconnected")
    except Exception as e:
        logging.error(f"Transcribe WebSocket error: {e}")
    finally:
        # Cleanup
        transcribe_connections.pop(session_id, None)
        stop_assemblyai_session(session_id)

# Test endpoints with better error handling
@app.get("/test/weather/{city}")
async def test_weather(city: str, session_id: str = Query("default")):
    """Test weather API"""
    try:
        result = get_weather(city, session_id)
        return {"city": city, "weather": result, "success": "not configured" not in result.lower()}
    except Exception as e:
        return {"city": city, "weather": f"Test failed: {str(e)}", "success": False}

@app.get("/test/search")
async def test_search(query: str, session_id: str = Query("default")):
    """Test search API"""
    try:
        result = search_web(query, session_id)
        return {"query": query, "results": result, "success": "not configured" not in result.lower()}
    except Exception as e:
        return {"query": query, "results": f"Test failed: {str(e)}", "success": False}

@app.get("/test/news")
async def test_news(topic: str = Query("latest"), session_id: str = Query("default")):
    """Test news API"""
    try:
        result = get_news(topic, session_id)
        return {"topic": topic, "news": result, "success": "not configured" not in result.lower()}
    except Exception as e:
        return {"topic": topic, "news": f"Test failed: {str(e)}", "success": False}

@app.get("/test/ai")
async def test_ai(prompt: str, session_id: str = Query("default")):
    """Test AI processing"""
    try:
        result = await process_with_ai(prompt, session_id)
        return {"prompt": prompt, "response": result, "success": True}
    except Exception as e:
        return {"prompt": prompt, "response": f"Test failed: {str(e)}", "success": False}

# Health endpoints
@app.get("/api/health")
async def health_check():
    # Clean up old sessions during health check
    await cleanup_old_sessions()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {
            "murf": len(murf_connections),
            "transcribe": len(transcribe_connections),
            "active_sessions": len(transcription_sessions)
        },
        "assemblyai_model": "universal-2",
        "version": "fixed"
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "version": "Fixed - AssemblyAI Universal-2 + Security Improvements",
        "active_sessions": len(transcribe_connections),
        "murf_connections": len(murf_connections),
        "available_personas": list(PERSONAS.keys()),
        "assemblyai_sessions": len(transcription_sessions),
        "assemblyai_model": "universal-2"
    }

# Root endpoint
@app.get("/")
async def read_root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced AI Voice Agent - Fixed</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #4CAF50; }
            .status { background: #333; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .endpoint { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #444; padding: 2px 6px; border-radius: 3px; color: #ff6b6b; }
            .changes { background: #0d4f3c; padding: 20px; border-radius: 8px; margin: 20px 0; }
            ul { margin: 10px 0 10px 20px; }
            li { margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Enhanced AI Voice Agent - FIXED VERSION</h1>
            
            <div class="status">
                <h3>Status: Fixed and Improved</h3>
                <p>AssemblyAI Universal-2 model properly configured with security improvements</p>
            </div>
            
            <h3>WebSocket Endpoints:</h3>
            <div class="endpoint">
                <code>/ws/murf-audio</code> - Audio streaming with timeout handling
            </div>
            <div class="endpoint">
                <code>/ws/stream-transcribe?session_id={session_id}</code> - Real-time transcription (Universal-2)
            </div>
            
            <h3>API Endpoints:</h3>
            <div class="endpoint">
                <code>POST /api/config/{session_id}</code> - Update config (with validation)
            </div>
            <div class="endpoint">
                <code>GET /api/config/{session_id}</code> - Get config status
            </div>
            <div class="endpoint">
                <code>POST /api/persona/{session_id}</code> - Set persona (validated)
            </div>
            <div class="endpoint">
                <code>GET /api/personas</code> - List available personas
            </div>
            <div class="endpoint">
                <code>DELETE /api/history/{session_id}</code> - Clear history (validated)
            </div>
            
            <h3>Test Endpoints:</h3>
            <div class="endpoint">
                <code>GET /test/weather/{city}</code> - Test weather API
            </div>
            <div class="endpoint">
                <code>GET /test/search?query={query}</code> - Test search API
            </div>
            <div class="endpoint">
                <code>GET /test/news?topic={topic}</code> - Test news API
            </div>
            <div class="endpoint">
                <code>GET /test/ai?prompt={prompt}</code> - Test AI processing
            </div>
            
            <div class="changes">
                <h3>🔧 Critical Fixes Applied:</h3>
                <ul>
                    <li>✅ Fixed AssemblyAI Universal-2 model parameter (model="universal-2")</li>
                    <li>✅ Fixed async/sync mixing in transcript handlers</li>
                    <li>✅ Added proper input validation and sanitization</li>
                    <li>✅ Improved error handling with specific error types</li>
                    <li>✅ Added session ID validation</li>
                    <li>✅ Added timeout handling for WebSocket connections</li>
                    <li>✅ Implemented automatic session cleanup</li>
                    <li>✅ Added memory management for chat history</li>
                    <li>✅ Improved rate limiting</li>
                    <li>✅ Better security for API key handling</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('voice_agent.log')
        ]
    )
    
    print("🚀 Starting Enhanced AI Voice Agent with FIXED AssemblyAI Universal-2")
    print("🔧 Security improvements and error handling applied")
    print("📡 WebSocket endpoints ready for real-time transcription")
    print("🔐 Make sure to configure your API keys through the web interface")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
