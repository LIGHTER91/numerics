import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle

def load_movies_dataset(movies_path, ratings_path, links_path):
    """
    Load and preprocess the movies dataset including tags.

    Args:
        movies_path (str): Path to the movies CSV file.
        ratings_path (str): Path to the ratings CSV file.
        links_path (str): Path to the links CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Preprocessed DataFrame containing movie information with tags
            - pd.DataFrame: Ratings DataFrame
    """
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    links_df = pd.read_csv(links_path)
    
    # Group tags by movieId and concatenate them
    tags_grouped = links_df.groupby('movieId')['tag'].agg(lambda x: ' '.join(x)).reset_index()
    # Merge tags with movies
    movies_df = movies_df.merge(tags_grouped, on='movieId', how='left')
    # Fill NaN tags with empty string
    movies_df['tag'] = movies_df['tag'].fillna('')
    
    # Create text field combining title, genres, and tags
    movies_df['text'] = movies_df.apply(
        lambda row: f"{row['genres']} {row['tag']}", axis=1
    )

    return movies_df, ratings_df

def generate_clip_embeddings(df, model, processor, batch_size=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate CLIP embeddings for text fields in the dataset using batch processing.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'text' column.
        model (CLIPModel): CLIP model.
        processor (CLIPProcessor): CLIP processor.
        batch_size (int): Number of texts to process in a batch.
        device (str): Device to perform computations ('cuda' or 'cpu').

    Returns:
        np.ndarray: Array of CLIP embeddings.
    """
    embeddings = []
    model = model.to(device)

    for i in tqdm(range(0, len(df), batch_size), desc="Generating embeddings"):
        batch_texts = df['text'].iloc[i:i + batch_size].tolist()
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs).cpu().numpy()
        embeddings.append(outputs)

    return np.vstack(embeddings)

def fit_knn(embeddings, n_neighbors=20):
    """
    Fit a k-NN model on the CLIP embeddings.

    Args:
        embeddings (np.ndarray): CLIP embeddings.
        n_neighbors (int): Number of neighbors for k-NN.

    Returns:
        NearestNeighbors: Trained k-NN model.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(embeddings)
    return knn

def save_model_and_embeddings(knn_model, embeddings, model_path, embeddings_path):
    """
    Save the trained k-NN model and embeddings to disk.

    Args:
        knn_model (NearestNeighbors): Trained k-NN model.
        embeddings (np.ndarray): CLIP embeddings.
        model_path (str): Path to save the k-NN model.
        embeddings_path (str): Path to save the embeddings.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(knn_model, f)
    np.save(embeddings_path, embeddings)

def load_model_and_embeddings(model_path, embeddings_path):
    """
    Load the trained k-NN model and embeddings from disk.

    Args:
        model_path (str): Path to the saved k-NN model.
        embeddings_path (str): Path to the saved embeddings.

    Returns:
        tuple: Loaded k-NN model and embeddings.
    """
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)
    embeddings = np.load(embeddings_path)
    return knn_model, embeddings

def generate_user_embeddings(ratings_df, movies_df, embeddings, min_rating=2.8):
    """
    Generate embeddings for users based on their liked movies.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing user ratings.
        movies_df (pd.DataFrame): DataFrame containing movie data.
        embeddings (np.ndarray): Precomputed embeddings for movies.
        min_rating (float): Minimum rating to consider a movie as liked by the user.

    Returns:
        dict: Dictionary mapping user IDs to their embeddings.
    """
    user_embeddings = {}
    for user_id in tqdm(ratings_df['userId'].unique(), desc="Generating user embeddings"):
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        liked_movies = user_ratings[user_ratings['rating'] >= min_rating]
        liked_movie_indices = movies_df[movies_df['movieId'].isin(liked_movies['movieId'])].index

        if len(liked_movie_indices) > 0:
            user_embeddings[user_id] = np.mean(embeddings[liked_movie_indices], axis=0)
        else:
            user_embeddings[user_id] = np.mean(embeddings, axis=0)

    return user_embeddings

def save_user_embeddings(user_embeddings, user_embeddings_path):
    """
    Save user embeddings to disk.

    Args:
        user_embeddings (dict): Dictionary of user embeddings.
        user_embeddings_path (str): Path to save the user embeddings.
    """
    with open(user_embeddings_path, 'wb') as f:
        pickle.dump(user_embeddings, f)

def load_user_embeddings(user_embeddings_path):
    """
    Load user embeddings from disk.

    Args:
        user_embeddings_path (str): Path to the saved user embeddings.

    Returns:
        dict: Dictionary of user embeddings.
    """
    with open(user_embeddings_path, 'rb') as f:
        return pickle.load(f)

def recommend_movies_for_user_with_embeddings(user_id, knn_model, movies_df, user_embeddings):
    """
    Recommend movies for a user using their precomputed embedding.

    Args:
        user_id (int): User ID for which to generate recommendations.
        knn_model (NearestNeighbors): Trained k-NN model.
        movies_df (pd.DataFrame): DataFrame containing movie data.
        user_embeddings (dict): Dictionary of user embeddings.

    Returns:
        pd.DataFrame: DataFrame containing recommended movies.
    """
    if user_id not in user_embeddings:
        print(f"No embedding found for user {user_id}.")
        return pd.DataFrame()

    user_embedding = user_embeddings[user_id].reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_embedding)
    recommended_indices = indices[0]

    return movies_df.iloc[recommended_indices]

def get_user_liked_movies(user_id, ratings_df, movies_df, min_rating=2.5):
    """
    Get movies that a user liked based on their ratings.

    Args:
        user_id (int): User ID to filter ratings.
        ratings_df (pd.DataFrame): DataFrame containing user ratings.
        movies_df (pd.DataFrame): DataFrame containing movie data.
        min_rating (float): Minimum rating to consider a movie as liked by the user.

    Returns:
        pd.DataFrame: DataFrame containing movies liked by the user with their ratings.
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= min_rating]
    liked_movies_with_details = movies_df[movies_df['movieId'].isin(liked_movies['movieId'])]
    
    # Merge with ratings to include the rating
    liked_movies_with_details = liked_movies_with_details.merge(
        user_ratings[['movieId', 'rating']], 
        on='movieId', 
        how='left'
    )
    
    return liked_movies_with_details
def recommend_movies_for_user_with_embeddings(user_id, knn_model, movies_df, user_embeddings, liked_movies_df=None):
    """
    Recommend movies for a user using their precomputed embedding, excluding already liked movies.
    
    Args:
        user_id (int): User ID for recommendations
        knn_model (NearestNeighbors): Trained k-NN model
        movies_df (pd.DataFrame): Movies data
        user_embeddings (dict): User embeddings dictionary
        liked_movies_df (pd.DataFrame): DataFrame of user's liked movies
        
    Returns:
        pd.DataFrame: Recommended movies
    
    Raises:
        ValueError: If all recommended movies were already liked
    """
    if user_id not in user_embeddings:
        print(f"No embedding found for user {user_id}.")
        return pd.DataFrame()

    user_embedding = user_embeddings[user_id].reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_embedding)
    recommended_df = movies_df.iloc[indices[0]]
    
    if liked_movies_df is not None:
        
        recommended_df = recommended_df[~recommended_df['movieId'].isin(liked_movies_df['movieId'])]
        
        if recommended_df.empty:
            raise ValueError("All recommended movies were already liked by the user")
            
    return recommended_df

def main():
    """
    Main function to execute the recommendation pipeline with user embeddings.
    """
    pd.set_option('display.max_colwidth', None)

    # Load the dataset
    movies_path = './data/movies.csv'
    ratings_path = './data/ratings.csv'
    links_path = './data/tags.csv'

    movies_df, ratings_df = load_movies_dataset(movies_path, ratings_path, links_path)

    # Define paths for saving and loading
    model_path = 'knn_model.pkl'
    embeddings_path = 'clip_embeddings.npy'
    user_embeddings_path = 'user_embeddings.pkl'

    # Check if saved model and embeddings exist
    try:
        knn_model, embeddings = load_model_and_embeddings(model_path, embeddings_path)
        print("Loaded saved model and embeddings.")
    except FileNotFoundError:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        embeddings = generate_clip_embeddings(movies_df, model, processor)
        knn_model = fit_knn(embeddings)

        save_model_and_embeddings(knn_model, embeddings, model_path, embeddings_path)
        print("Model and embeddings saved.")

    # Check if user embeddings exist
    try:
        user_embeddings = load_user_embeddings(user_embeddings_path)
        print("Loaded saved user embeddings.")
    except FileNotFoundError:
        user_embeddings = generate_user_embeddings(ratings_df, movies_df, embeddings)
        save_user_embeddings(user_embeddings, user_embeddings_path)
        print("User embeddings saved.")

    # User ID for recommendation
    user_id = 1# 506
    liked_movies = get_user_liked_movies(user_id, ratings_df, movies_df)
    liked_movies = liked_movies.sort_values('rating', ascending=False)
    print(f"Movies liked by user {user_id}:")
    print(liked_movies[['title', 'genres','rating']].head(200))
    
    try:
        recommendations = recommend_movies_for_user_with_embeddings(
            user_id, 
            knn_model, 
            movies_df, 
            user_embeddings, 
            liked_movies
        )
        print(f"Recommendations for user {user_id}:")
        print(recommendations[['title', 'genres']])
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
