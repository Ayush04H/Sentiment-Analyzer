from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

app = FastAPI()

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the data model for the input
class SentimentRequest(BaseModel):
    text: str

# Utility function to analyze sentiment
def analyze_sentiment(text: str) -> dict:
    return sia.polarity_scores(text)

# Utility function to create a bar graph of sentiment scores
def create_sentiment_graph(scores: dict) -> io.BytesIO:
    categories = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(8, 4))
    plt.bar(categories, values, color=['red', 'blue', 'green', 'purple'])
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Score')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img

@app.post("/analyze/")
async def analyze_text(request: SentimentRequest):
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Analyze the sentiment
    sentiment_scores = analyze_sentiment(text)

    # Return sentiment scores as JSON
    return JSONResponse(content={
        "text": text,
        "sentiment": sentiment_scores
    })

@app.get("/graph/")
async def get_sentiment_graph(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Analyze the sentiment
    sentiment_scores = analyze_sentiment(text)

    # Generate the sentiment graph
    sentiment_graph = create_sentiment_graph(sentiment_scores)

    # Return the graph image
    return StreamingResponse(sentiment_graph, media_type="image/png")

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ for sentiment scores and /graph/ to get a sentiment graph."}
