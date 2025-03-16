import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv('cleaned_dataset.csv')
print("Dataset Loaded Successfully ✅")

# VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

data['VADER Sentiment'] = data['Cleaned_Post'].apply(get_vader_sentiment)

# TextBlob Emotion Analysis
def get_emotion(text):
    analysis = TextBlob(text).sentiment
    if analysis.polarity > 0.3:
        return 'Joy'
    elif analysis.polarity < -0.3:
        return 'Sadness'
    else:
        return 'Neutral'

data['Emotion'] = data['Cleaned_Post'].apply(get_emotion)

# Display and Save Results
print("Final Data with Emotions ✅")
print(data[['Cleaned_Post', 'VADER Sentiment', 'Emotion']])
data.to_csv("emotion_results.csv", index=False)
print("Results saved successfully ✅")

# Interactive Input
while True:
    user_input = input("\nEnter a social media post (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting... 👋")
        break
    print(f"Predicted Emotion: {get_emotion(user_input)}")
    print(f"Predicted Sentiment: {get_vader_sentiment(user_input)}")
