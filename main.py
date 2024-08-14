from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import analyze_sentiment, create_sentiment_graph

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

app = FastAPI()

# Define the data model for the input
class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    sentiment_scores = analyze_sentiment(text)
    return JSONResponse(content={
        "text": text,
        "sentiment": sentiment_scores
    })

@app.get("/graph/")
async def get_sentiment_graph(text: str = Query(..., description="Text to analyze")):
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    sentiment_scores = analyze_sentiment(text)
    sentiment_graph = create_sentiment_graph(sentiment_scores)
    return StreamingResponse(sentiment_graph, media_type="image/png")

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores and /graph/?text=TEXT to get a sentiment graph."}
