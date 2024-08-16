import matplotlib.pyplot as plt
import seaborn as sns
import io
from pymongo import MongoClient
from bson import ObjectId
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from logger.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_db"]
texts_collection = db["texts"]
images_collection = db["images"]

def analyze_sentiment(text: str):
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        logger.info(f"Sentiment analysis complete for text: {text}")
        return sentiment_scores
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise

def create_sentiment_graph(sentiment_scores):
    try:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(sentiment_scores.keys()), y=list(sentiment_scores.values()))
        plt.title('Sentiment Scores')
        plt.xlabel('Sentiment Type')
        plt.ylabel('Score')
        plt.ylim(-1, 1)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        logger.info("Sentiment graph created successfully.")
        return buf
    except Exception as e:
        logger.error(f"Error creating sentiment graph: {str(e)}")
        raise

def save_text_to_db(text: str, sentiment_scores: dict):
    try:
        text_doc = {
            "text": text,
            "sentiment_scores": sentiment_scores
        }
        result = texts_collection.insert_one(text_doc)
        logger.info(f"Text and sentiment scores saved to MongoDB for text: {text}")
        return result
    except Exception as e:
        logger.error(f"Error saving text to MongoDB: {str(e)}")
        raise

def save_image_to_db(text_id: str, image_bytes: bytes):
    try:
        image_doc = {
            "text_id": text_id,
            "image": image_bytes
        }
        images_collection.insert_one(image_doc)
        logger.info(f"Sentiment graph image saved to MongoDB for text_id: {text_id}")
    except Exception as e:
        logger.error(f"Error saving image to MongoDB: {str(e)}")
        raise

def get_sentiment_scores_from_db(text_id: str):
    try:
        text_doc = texts_collection.find_one({"_id": ObjectId(text_id)})
        if not text_doc:
            raise ValueError("Text not found.")
        logger.info(f"Sentiment scores retrieved for text_id: {text_id}")
        return text_doc.get("sentiment_scores")
    except Exception as e:
        logger.error(f"Error retrieving sentiment scores from MongoDB: {str(e)}")
        raise

def get_dashboard_data():
    try:
        texts = list(texts_collection.find())
        
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
        
        logger.info("Dashboard data aggregated successfully.")
        return aggregate_scores
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        raise
