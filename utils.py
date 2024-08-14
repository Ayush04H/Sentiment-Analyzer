import matplotlib.pyplot as plt
import seaborn as sns
import io
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    return sia.polarity_scores(text)

def create_sentiment_graph(scores: dict) -> io.BytesIO:
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", 4)
    
    categories = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=palette, edgecolor='black')

    # Add value annotations on each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.02, 
            f'{height:.2f}', 
            ha='center', 
            va='bottom',
            fontsize=12
        )

    plt.title('Sentiment Analysis Results', fontsize=16)
    plt.xlabel('Sentiment Type', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return img
