import matplotlib.pyplot as plt
import seaborn as sns
import io
from pymongo import MongoClient
from bson import ObjectId
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_db"]
texts_collection = db["texts"]
images_collection = db["images"]

def analyze_sentiment(text: str):
    # Sentiment analysis logic
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def create_sentiment_graph(sentiment_scores):
    # Create a sentiment graph
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(sentiment_scores.keys()), y=list(sentiment_scores.values()))
    plt.title('Sentiment Scores')
    plt.xlabel('Sentiment Type')
    plt.ylabel('Score')
    plt.ylim(-1, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def save_text_to_db(text: str, sentiment_scores: dict):
    # Store text and sentiment scores in MongoDB
    text_doc = {
        "text": text,
        "sentiment_scores": sentiment_scores
    }
    return texts_collection.insert_one(text_doc)

def save_image_to_db(text_id: str, image_bytes: bytes):
    # Store the image in MongoDB
    image_doc = {
        "text_id": text_id,
        "image": image_bytes
    }
    images_collection.insert_one(image_doc)

def get_sentiment_scores_from_db(text_id: str):
    # Retrieve sentiment scores from MongoDB
    text_doc = texts_collection.find_one({"_id": ObjectId(text_id)})
    if not text_doc:
        raise ValueError("Text not found.")
    return text_doc.get("sentiment_scores")

def get_dashboard_data():
    # Retrieve all sentiment scores from MongoDB
    texts = list(texts_collection.find())
    
    # Aggregate sentiment scores
    aggregate_scores = {
        "neg": 0,
        "neu": 0,
        "pos": 0
    }
    count = len(texts)
    
    if count == 0:
        return aggregate_scores
    
    for text in texts:
        sentiment_scores = text.get("sentiment_scores")
        if sentiment_scores:
            for key in aggregate_scores.keys():
                aggregate_scores[key] += sentiment_scores.get(key, 0)
    
    for key in aggregate_scores.keys():
        aggregate_scores[key] /= count
    
    return aggregate_scores
