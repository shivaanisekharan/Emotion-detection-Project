import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

print("All libraries imported successfully ✅")

# Sample Data (For Practice)
data = pd.DataFrame({
    'Post': [
        "I'm feeling amazing today! Everything is perfect.",
        "I'm so stressed and anxious, nothing feels right.",
        "Life is tough, but I'm pushing through.",
        "I feel empty and lost all the time...",
        "Wow, today was absolutely fantastic!",
        "I hate everything, I can’t stand this pain anymore.",
        "Today was decent, not great but manageable.",
        "I'm exhausted... nothing makes sense these days."
    ]
})

print("Sample Data Loaded Successfully ✅")
print(data)

import re

# Text Cleaning Function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)      # Remove mentions
    text = re.sub(r'#\w+', '', text)      # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower().strip()

# Clean the text
data['Cleaned_Post'] = data['Post'].apply(clean_text)

print("Cleaned Data ✅")
print(data[['Post', 'Cleaned_Post']])

# Save Cleaned Data to CSV
data.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved successfully ✅")

