# models/sentiment.py
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

def vader_sentiment(reviews_df):
    """Fast rule-based sentiment with VADER"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_score(text):
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']  # -1 to +1
    
    reviews_df['vader_score'] = reviews_df['review'].apply(get_score)
    print(reviews_df[['review', 'vader_score']].head())
    return reviews_df

def distilbert_sentiment(reviews_df, batch_size=32):
    """Deep learning sentiment with DistilBERT"""
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    
    reviews = reviews_df['review'].tolist()
    results = []
    
    # Process in batches
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        preds = classifier(batch)
        results.extend(preds)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(reviews)}")
    
    reviews_df['bert_label'] = [r['label'] for r in results]
    reviews_df['bert_score']  = [
        r['score'] if r['label'] == 'POSITIVE' else -r['score']
        for r in results
    ]
    return reviews_df

def aggregate_movie_sentiment(reviews_df):
    """Compute per-movie average sentiment score"""
    movie_sentiment = reviews_df.groupby('movieId').agg(
        avg_vader=('vader_score', 'mean'),
        avg_bert=('bert_score', 'mean'),
        review_count=('review', 'count')
    ).reset_index()
    
    # Combine both scores
    movie_sentiment['final_sentiment'] = (
        0.4 * movie_sentiment['avg_vader'] +
        0.6 * movie_sentiment['avg_bert']
    )
    movie_sentiment.to_csv('data/processed/movie_sentiment.csv', index=False)
    return movie_sentiment