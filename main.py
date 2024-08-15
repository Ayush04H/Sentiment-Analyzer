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

# In-memory storage to keep track of the last analyzed text and its sentiment scores
last_analyzed = {
    "text": None,
    "sentiment_scores": None
}

@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    sentiment_scores = analyze_sentiment(text)
    
    # Store the analyzed text and sentiment scores
    last_analyzed["text"] = text
    last_analyzed["sentiment_scores"] = sentiment_scores
    
    return JSONResponse(content={
        "text": text,
        "sentiment": sentiment_scores
    })

@app.get("/graph/")
async def get_sentiment_graph():
    if not last_analyzed["text"]:
        raise HTTPException(status_code=400, detail="No text has been analyzed yet.")
    
    sentiment_scores = last_analyzed["sentiment_scores"]
    sentiment_graph = create_sentiment_graph(sentiment_scores)
    
    return StreamingResponse(sentiment_graph, media_type="image/png")

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores and /graph/ to get a sentiment graph."}
