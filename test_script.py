#!/usr/bin/env python
# Test script for Twitter Sentiment Analysis

# disable warning
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

print("Loading dataset...")
df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv', header=None, index_col=[0])
df = df[[2,3]].reset_index(drop=True)
df.columns = ['sentiment', 'text']
print(f"Dataset loaded with shape: {df.shape}")
print(df.head())

print("\nDataset info:")
df.info()

df.isnull().sum()

df.dropna(inplace=True)
df = df[df['text'].apply(len)>1]

print("\nSentiment value counts:")
print(df['sentiment'].value_counts())

# basic feature extraction | 
import preprocess_kgptalkie as ps

print("\nExtracting basic features manually...")
df['word_count'] = df['text'].apply(lambda x: ps.word_count(x))
df['char_count'] = df['text'].apply(lambda x: ps.char_count(x))
df['avg_word_len'] = df['text'].apply(lambda x: ps.avg_word_len(x))
df['stop_words_count'] = df['text'].apply(lambda x: ps.stop_words_count(x))

print(f"DataFrame columns after feature extraction: {list(df.columns)}")

# train test split
from sklearn.model_selection import train_test_split

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# model building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

print("\nTraining model...")
clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')), 
    ('clf', RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42))
])

clf.fit(X_train, y_train)

# evaluation
from sklearn.metrics import accuracy_score

print("\nEvaluating model...")
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# save model
import pickle

print("Saving model...")
pickle.dump(clf, open('twitter_sentiment.pkl', 'wb'))
print("Model saved as twitter_sentiment.pkl")

print("Script completed successfully!")