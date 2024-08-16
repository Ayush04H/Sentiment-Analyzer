from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from utils import analyze_sentiment, create_sentiment_graph, save_text_to_db, save_image_to_db, get_sentiment_scores_from_db, get_dashboard_data
from logger.logger import get_logger
import io

app = FastAPI()

# Initialize logger
logger = get_logger(__name__)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_db"]
texts_collection = db["texts"]
images_collection = db["images"]

# Define the data model for the input
class SentimentRequest(BaseModel):
    text: str
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores, /graph/?text_id=ID to get a sentiment graph, and /dashboard/ to get a sentiment dashboard."}


@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    try:
        text = request.text
        if not text:
            raise HTTPException(status_code=400, detail="Text input is required.")
        
        sentiment_scores = analyze_sentiment(text)
        
        # Store text and sentiment scores in MongoDB
        result = save_text_to_db(text, sentiment_scores)
        
        logger.info(f"Sentiment analysis successful for text: {text}")
        return JSONResponse(content={
            "text": text,
            "sentiment": sentiment_scores,
            "id": str(result.inserted_id)
        })
    except Exception as e:
        logger.error(f"Error in /analyze/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during sentiment analysis.")

@app.get("/graph/")
async def get_sentiment_graph(text_id: str = Query(..., description="Text ID to generate graph")):
    try:
        if not text_id:
            raise HTTPException(status_code=400, detail="Text ID is required.")
        
        # Retrieve the sentiment scores from MongoDB
        sentiment_scores = get_sentiment_scores_from_db(text_id)
        
        sentiment_graph = create_sentiment_graph(sentiment_scores)
        
        # Store the image in MongoDB
        save_image_to_db(text_id, sentiment_graph.getvalue())
        
        logger.info(f"Sentiment graph generated for text_id: {text_id}")
        return StreamingResponse(io.BytesIO(sentiment_graph.getvalue()), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /graph/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the sentiment graph.")

@app.get("/dashboard/")
async def get_dashboard():
    try:
        # Retrieve all sentiment scores from MongoDB
        aggregate_scores = get_dashboard_data()
        
        if not aggregate_scores:
            raise HTTPException(status_code=404, detail="No sentiment data available.")
        
        dashboard_graph = create_sentiment_graph(aggregate_scores)
        
        logger.info("Dashboard data retrieved and graph generated.")
        return StreamingResponse(io.BytesIO(dashboard_graph.getvalue()), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /dashboard/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the dashboard.")

