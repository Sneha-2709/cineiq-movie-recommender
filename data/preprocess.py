# data/preprocess.py
import pandas as pd
import ast

def clean_ratings(ratings):
    """Filter users/movies with too few ratings"""
    # Keep movies with >= 50 ratings
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(
        movie_counts[movie_counts >= 50].index
    )]
    # Keep users with >= 20 ratings
    user_counts = ratings['userId'].value_counts()
    ratings = ratings[ratings['userId'].isin(
        user_counts[user_counts >= 20].index
    )]
    print(f"Filtered ratings: {ratings.shape}")
    return ratings

def extract_genres(movies):
    """One-hot encode genres"""
    genres_dummies = movies['genres'].str.get_dummies('|')
    movies = pd.concat([movies, genres_dummies], axis=1)
    return movies

def extract_cast_crew(credits):
    """Parse JSON-like cast and director columns"""
    credits['cast'] = credits['cast'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)[:3]]
    )
    credits['director'] = credits['crew'].apply(
        lambda x: next(
            (i['name'] for i in ast.literal_eval(x) 
             if i['job'] == 'Director'), None
        )
    )
    return credits[['movie_id', 'cast', 'director']]

def build_content_features(movies_meta, credits):
    """Combine overview, genres, cast, director into one text blob"""
    df = movies_meta.merge(credits, left_on='id', right_on='movie_id')
    df['soup'] = (
        df['overview'].fillna('') + ' ' +
        df['genres'].fillna('') + ' ' +
        df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
        df['director'].fillna('')
    )
    df.to_csv('data/processed/content_features.csv', index=False)
    return df
