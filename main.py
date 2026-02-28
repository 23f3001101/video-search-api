from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import subprocess
import tempfile
import time
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class VideoRequest(BaseModel):
    video_url: str
    topic: str

@app.post("/ask")
async def ask(request: VideoRequest):
    audio_path = None
    uploaded_file = None

    try:
        # Step 1: Download audio only using yt-dlp
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name

        subprocess.run([
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", audio_path,
            "--no-playlist",
            "--js-runtimes", "nodejs",
            request.video_url
        ], check=True, capture_output=True)      
        
        # Step 2: Upload to Gemini Files API
        uploaded_file = genai.upload_file(
            path=audio_path,
            mime_type="audio/mpeg"
        )

        # Step 3: Poll until file is ACTIVE
        max_wait = 120
        waited = 0
        while uploaded_file.state.name != "ACTIVE":
            if waited >= max_wait:
                raise HTTPException(status_code=500, detail="File processing timed out")
            time.sleep(5)
            waited += 5
            uploaded_file = genai.get_file(uploaded_file.name)

        # Step 4: Ask Gemini to find the timestamp
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Listen to this audio and find the exact timestamp when the following topic is first spoken or discussed:

Topic: "{request.topic}"

Return ONLY a JSON object in this exact format:
{{"timestamp": "HH:MM:SS"}}

Rules:
- timestamp must be in HH:MM:SS format (e.g. "00:05:47", "01:23:45")
- Return the first occurrence only
- If not found, return "00:00:00"
"""

        response = model.generate_content(
            [uploaded_file, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        import json
        result = json.loads(response.text)
        timestamp = result.get("timestamp", "00:00:00")

        # Validate HH:MM:SS format
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", timestamp):
            timestamp = "00:00:00"

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"yt-dlp error: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
