from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

@app.post("/ask")
async def ask(request: VideoRequest):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Watch this YouTube video and find the FIRST timestamp where the following topic is spoken or discussed:

Topic: "{request.topic}"

Return ONLY a JSON object:
{{"timestamp": "HH:MM:SS"}}

Rules:
- Use exact HH:MM:SS format (e.g. "00:05:47")
- Return the first occurrence only
- If not found, return "00:00:00"
"""

        response = model.generate_content(
            [
                {
                    "file_data": {
                        "mime_type": "video/youtube",
                        "file_uri": request.video_url
                    }
                },
                prompt
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
