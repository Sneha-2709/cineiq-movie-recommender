import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import pickle, os, mlflow

os.makedirs("models/saved", exist_ok=True)

df = pd.read_csv("data/processed/ratings_clean.csv")

user_enc = LabelEncoder()
movie_enc = LabelEncoder()
df["user_idx"] = user_enc.fit_transform(df["userId"])
df["movie_idx"] = movie_enc.fit_transform(df["movieId"])

n_users = df["user_idx"].nunique()
n_movies = df["movie_idx"].nunique()
matrix = np.zeros((n_users, n_movies))
for row in df.itertuples():
    matrix[row.user_idx, row.movie_idx] = row.rating

svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(matrix)

with mlflow.start_run(run_name="collaborative_svd"):
    mlflow.log_param("n_components", 100)
    mlflow.log_metric("explained_variance", float(svd.explained_variance_ratio_.sum()))

with open("models/saved/svd_model.pkl", "wb") as f:
    pickle.dump({"svd": svd, "matrix": matrix,
                 "user_enc": user_enc, "movie_enc": movie_enc}, f)

print("Collaborative model trained and saved successfully.")
