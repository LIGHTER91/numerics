import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle

def load_movies_dataset_with_genres(movies_path, links_path):
    """
    Load and preprocess the movies dataset including tags.

    Args:
        movies_path (str): Path to the movies CSV file.
        links_path (str): Path to the links CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing movie information with tags
    """
    movies_df = pd.read_csv(movies_path)
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

    return movies_df

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

def recommend_movies_by_genre(genres, knn_model, movies_df, embeddings):
    """
    Recommend movies based on the input genres.

    Args:
        genres (str): Comma-separated genres to filter movies.
        knn_model (NearestNeighbors): Trained k-NN model.
        movies_df (pd.DataFrame): DataFrame containing movie data.
        embeddings (np.ndarray): Precomputed embeddings for movies.

    Returns:
        pd.DataFrame: DataFrame containing recommended movies.
    """
    # Filter movies matching the genres
    genre_movies = movies_df[movies_df['genres'].str.contains(genres, case=False, na=False)]

    if genre_movies.empty:
        print(f"No movies found for genres: {genres}")
        return pd.DataFrame()

    # Compute mean embedding for the filtered movies
    genre_indices = genre_movies.index
    genre_embedding = np.mean(embeddings[genre_indices], axis=0).reshape(1, -1)

    # Find similar movies
    distances, indices = knn_model.kneighbors(genre_embedding)
    recommended_indices = indices[0]

    return movies_df.iloc[recommended_indices]

def get_recomandation(genres):
    """
    Main function to execute the recommendation pipeline based on genres.
    """
    pd.set_option('display.max_colwidth', None)

    # Load the dataset
    movies_path = './data/movies.csv'
    links_path = './data/tags.csv'

    movies_df = load_movies_dataset_with_genres(movies_path, links_path)

    # Define paths for saving and loading
    model_path = 'knn_model.pkl'
    embeddings_path = 'clip_embeddings.npy'

    # Check if saved model and embeddings exist
    try:
        with open(model_path, 'rb') as f:
            knn_model = pickle.load(f)
        embeddings = np.load(embeddings_path)
        print("Loaded saved model and embeddings.")
    except FileNotFoundError:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        embeddings = generate_clip_embeddings(movies_df, model, processor)
        knn_model = fit_knn(embeddings)

        with open(model_path, 'wb') as f:
            pickle.dump(knn_model, f)
        np.save(embeddings_path, embeddings)
        print("Model and embeddings saved.")


    recommendations = recommend_movies_by_genre(genres, knn_model, movies_df, embeddings)
    return recommendations