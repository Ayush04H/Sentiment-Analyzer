from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from utils import analyze_sentiment, create_sentiment_graph, save_text_to_db, save_image_to_db, get_sentiment_scores_from_db, get_dashboard_data, create_dashboard_graph
import io
import pandas as pd
from logger.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

app = FastAPI()

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
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores, /graph/?text_id=ID to get a sentiment graph, and /dashboard/ to get a sentiment dashboard."}

@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    try:
        text = request.text
        if not text:
            logger.warning("No text input provided.")
            raise HTTPException(status_code=400, detail="Text input is required.")
        
        sentiment_scores = analyze_sentiment(text)
        
        # Store text and sentiment scores in MongoDB
        result = save_text_to_db(text, sentiment_scores)
        
        logger.info("Sentiment analysis and data storage successful.")
        return JSONResponse(content={
            "text": text,
            "sentiment": sentiment_scores,
            "id": str(result.inserted_id)
        })
    except Exception as e:
        logger.error(f"Error in /analyze/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing the text.")

@app.get("/graph/")
async def get_sentiment_graph(text_id: str = Query(..., description="Text ID to generate graph")):
    try:
        if not text_id:
            logger.warning("No text ID provided for graph generation.")
            raise HTTPException(status_code=400, detail="Text ID is required.")
        
        # Retrieve the sentiment scores from MongoDB
        sentiment_scores = get_sentiment_scores_from_db(text_id)
        
        sentiment_graph = create_sentiment_graph(sentiment_scores)
        
        # Store the image in MongoDB
        save_image_to_db(text_id, sentiment_graph.getvalue())
        
        logger.info("Sentiment graph generated and stored successfully.")
        return StreamingResponse(io.BytesIO(sentiment_graph.getvalue()), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /graph/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the graph.")

@app.get("/dashboard/")
async def get_dashboard():
    try:
        # Retrieve all sentiment scores from MongoDB
        aggregate_scores = get_dashboard_data()
        
        if not aggregate_scores:
            logger.warning("No sentiment data available for dashboard.")
            raise HTTPException(status_code=404, detail="No sentiment data available.")
        
        dashboard_graph = create_dashboard_graph(aggregate_scores)
        
        logger.info("Dashboard data retrieved and enhanced graph generated.")
        return StreamingResponse(io.BytesIO(dashboard_graph.getvalue()), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /dashboard/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the dashboard.")

@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        contents = await file.read()
        if not contents.strip():  # Check if the file is empty
            logger.warning("Uploaded CSV file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        
        # Try reading the CSV file with different encodings
        df = None
        for encoding in ['utf-8', 'ISO-8859-1']:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                if df.empty:  # Check if the DataFrame is empty
                    logger.warning("Uploaded CSV file has no data.")
                    raise HTTPException(status_code=400, detail="Uploaded CSV file has no data.")
                break  # If reading is successful, break out of the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
            except pd.errors.EmptyDataError as e:
                logger.error(f"CSV parsing error: {str(e)}")
                raise HTTPException(status_code=400, detail="CSV file is empty or improperly formatted.")
            except pd.errors.ParserError as e:
                logger.error(f"CSV parsing error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
        
        if df is None:
            logger.error("Unable to decode the CSV file with any supported encoding.")
            raise HTTPException(status_code=400, detail="Unable to decode the CSV file with any supported encoding.")
        
        # Check if the required 'Sentiments' column is present
        if 'Sentiments' not in df.columns:
            logger.warning("Uploaded CSV does not contain 'Sentiments' column.")
            raise HTTPException(status_code=400, detail="CSV must contain a 'Sentiments' column.")
        
        # Extract the sentiments from the CSV
        sentiments = df['Sentiments'].dropna().tolist()  # Drop any rows with NaN values
        
        if not sentiments:  # Check if there are any sentiments to process
            logger.warning("No valid sentiment data found in CSV.")
            raise HTTPException(status_code=400, detail="CSV contains no valid sentiment data.")
        
        # Perform sentiment analysis for each sentiment in the list
        results = []
        for sentiment in sentiments:
            sentiment_scores = analyze_sentiment(sentiment)
            result = save_text_to_db(sentiment, sentiment_scores)
            results.append({
                "id": str(result.inserted_id),
                "text": sentiment,
                "neg": sentiment_scores["neg"],
                "neu": sentiment_scores["neu"],
                "pos": sentiment_scores["pos"],
                "compound": sentiment_scores["compound"]
            })
        
        # Convert results to a DataFrame and prepare CSV response
        result_df = pd.DataFrame(results)
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        
        logger.info("CSV sentiment analysis completed successfully.")
        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=result_{file.filename}"})
    except Exception as e:
        logger.error(f"Error in /analyze-csv/: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the CSV.")
