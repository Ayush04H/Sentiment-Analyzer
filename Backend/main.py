from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from utils import analyze_sentiment, create_sentiment_graph, save_text_to_db, save_image_to_db, get_sentiment_scores_from_db, get_dashboard_data
import io
app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_db"]
texts_collection = db["texts"]
images_collection = db["images"]

# Define the data model for the input
class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    sentiment_scores = analyze_sentiment(text)
    
    # Store text and sentiment scores in MongoDB
    result = save_text_to_db(text, sentiment_scores)
    
    return JSONResponse(content={
        "text": text,
        "sentiment": sentiment_scores,
        "id": str(result.inserted_id)
    })

@app.get("/graph/")
async def get_sentiment_graph(text_id: str = Query(..., description="Text ID to generate graph")):
    if not text_id:
        raise HTTPException(status_code=400, detail="Text ID is required.")
    
    # Retrieve the sentiment scores from MongoDB
    sentiment_scores = get_sentiment_scores_from_db(text_id)
    
    sentiment_graph = create_sentiment_graph(sentiment_scores)
    
    # Store the image in MongoDB
    save_image_to_db(text_id, sentiment_graph.getvalue())
    
    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(sentiment_graph.getvalue()), media_type="image/png")

@app.get("/dashboard/")
async def get_dashboard():
    # Retrieve all sentiment scores from MongoDB
    aggregate_scores = get_dashboard_data()
    
    if not aggregate_scores:
        raise HTTPException(status_code=404, detail="No sentiment data available.")
    
    dashboard_graph = create_sentiment_graph(aggregate_scores)
    
    return StreamingResponse(io.BytesIO(dashboard_graph.getvalue()), media_type="image/png")

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores, /graph/?text_id=ID to get a sentiment graph, and /dashboard/ to get a sentiment dashboard."}
