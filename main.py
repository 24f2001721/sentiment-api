from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ✅ Only difference — point to AIPipe instead of OpenAI
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the user's comment. "
                        "Return 'positive', 'negative', or 'neutral' for sentiment. "
                        "Return a rating from 1 (very negative) to 5 (very positive)."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4, 5]
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return SentimentResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
