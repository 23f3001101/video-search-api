from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import os, re, json
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

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

@app.post("/ask")
async def ask(request: VideoRequest):
    try:
        # Step 1: Extract video ID
        video_id = extract_video_id(request.video_url)

        # Step 2: Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Step 3: Format transcript with timestamps
        transcript_text = ""
        for entry in transcript:
            ts = seconds_to_hhmmss(entry["start"])
            transcript_text += f"[{ts}] {entry['text']}\n"

        # Step 4: Ask Gemini to find the timestamp
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Here is a YouTube video transcript with timestamps in [HH:MM:SS] format.

Find the FIRST timestamp where the following topic is spoken or discussed:
Topic: "{request.topic}"

TRANSCRIPT:
{transcript_text[:50000]}

Return ONLY a JSON object:
{{"timestamp": "HH:MM:SS"}}

Rules:
- Use exact HH:MM:SS format (e.g. "00:05:47")
- Return the first occurrence
- If not found, return "00:00:00"
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        result = json.loads(response.text)
        timestamp = result.get("timestamp", "00:00:00")

        # Validate format
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", timestamp):
            timestamp = "00:00:00"

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
