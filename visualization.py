import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Load the emotion results dataset
data = pd.read_csv("emotion_results.csv")

# Visualizing Sentiment Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x='VADER Sentiment', palette='viridis')
plt.title("Distribution of Sentiments", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Visualizing Emotion Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x='Emotion', palette='coolwarm')
plt.title("Distribution of Emotions", fontsize=14)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()

# Interactive Pie Chart for Emotions
fig = px.pie(data, names='Emotion', title='Emotion Distribution (Interactive)', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

# Sentiment vs Emotion Analysis Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(data['VADER Sentiment'], data['Emotion']), annot=True, cmap='YlGnBu', fmt='d')
plt.title("Sentiment vs Emotion Analysis", fontsize=14)
plt.show()
