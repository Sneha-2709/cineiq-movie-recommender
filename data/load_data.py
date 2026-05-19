# data/load_data.py
import pandas as pd

def load_movielens():
    """Load MovieLens 25M dataset"""
    # Download from: https://grouplens.org/datasets/movielens/25m/
    # Place files in data/raw/

    ratings = pd.read_csv('data/raw/ratings.csv')
    movies  = pd.read_csv('data/raw/movies.csv')
    
    print(f"Ratings shape: {ratings.shape}")
    print(f"Movies shape:  {movies.shape}")
    print(ratings.head())
    return ratings, movies

def load_tmdb():
    """Load TMDB metadata from Kaggle"""
    # Download: kaggle datasets download tmdb-movie-metadata
    movies_meta = pd.read_csv('data/raw/tmdb_5000_movies.csv')
    credits      = pd.read_csv('data/raw/tmdb_5000_credits.csv')
    return movies_meta, credits

def load_imdb_reviews():
    """Load IMDB 50K reviews for sentiment training"""
    # Download: kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    reviews = pd.read_csv('data/raw/IMDB Dataset.csv')
    print(reviews['sentiment'].value_counts())
    return reviews

if __name__ == "__main__":
    ratings, movies = load_movielens()
    reviews = load_imdb_reviews()