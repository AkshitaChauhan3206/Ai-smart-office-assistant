import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

print("Generating synthetic dataset for Spam/Priority classifier...")

# Create a synthetic dataset
data = {
    "text": [
        "URGENT: Your account has been suspended. Click here to verify now!",
        "WIN $1,000,000 risk-free! Claim your prize today",
        "Congratulations! You have been selected for a free gift card.",
        "Limited time offer! Act now to secure your jackpot.",
        "Meeting at 10 AM tomorrow to discuss the Q3 roadmap.",
        "Please review the attached quarterly presentation by Friday.",
        "Can we reschedule our 1:1 for next week? I have a conflict.",
        "Following up on the deploy status for the v2.0 release.",
        "CRITICAL: Production server is down, please investigate ASAP.",
        "URGENT: Executive summary needed by 5 PM today.",
        "Hey, do you want to grab lunch around noon?",
        "See you at the team building event on Friday!",
        "Just wanted to say thanks for your help yesterday.",
        "Free money guaranteed!",
        "Click here to instantly win a lottery.",
        "Important update regarding your healthcare benefits for next year.",
        "Action Required: Please sign the new employee handbook.",
        "Reminder: All hands meeting in 15 minutes."
    ]*5, # multiply to get enough rows
    "label": [
        "Spam", "Spam", "Spam", "Spam",
        "Normal", "Normal", "Normal", "Normal",
        "Important", "Important",
        "Normal", "Normal", "Normal",
        "Spam", "Spam",
        "Important", "Important", "Normal"
    ]*5
}

df = pd.DataFrame(data)

print("Training Naive Bayes Model with TF-IDF...")

# 1. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 2. Train Model
model = MultinomialNB()
model.fit(X, y)

# 3. Test accuracy
accuracy = model.score(X, y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# 4. Save to disk
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Models saved as vectorizer.pkl and spam_model.pkl successfully!")