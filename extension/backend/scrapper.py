import requests
import requests
from bs4 import BeautifulSoup
# Clé API et URL de base
API_KEY = "addf1ec8d43e8e75ebf20270f8132652"  # Remplacez par votre clé API
BASE_URL = "https://api.themoviedb.org/3"

def search_movie_by_name(movie_name):
    """
    Recherche un film par son nom via l'API TMDB.
    Retourne les informations du film, y compris ses genres.
    """
    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_name,
        "language": "en-US"  
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            return data["results"]  # Liste de films correspondants
        else:
            print("Aucun film trouvé avec ce nom.")
            return []
    else:
        raise Exception(f"Erreur : {response.status_code}, {response.text}")

def get_genres_map():
    """
    Récupère la liste des genres disponibles via l'API TMDB.
    Retourne un dictionnaire {id_genre: nom_genre}.
    """
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY, "language": "en-US"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        genres = response.json()["genres"]
        return {genre["id"]: genre["name"] for genre in genres}
    else:
        raise Exception(f"Erreur : {response.status_code}, {response.text}")
def transform_genres(genres_list):
    """
    Transform a list of genres into a single string separated by '|'.

    Args:
        genres_list (list): List of genres.

    Returns:
        str: Formatted string of genres.
    """
    flattened_genres = {genre for sublist in genres_list for genre in sublist}
    return '|'.join(sorted(flattened_genres))

def get_profil_moovie(username):
    # URL du profil TMDB (par exemple, les films notés d'un utilisateur)
    url = f"https://www.themoviedb.org/u/{username}/ratings"
    
    # Envoyer une requête GET
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    genres_movies=[]
    # Vérifier si la requête est réussie
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Trouver tous les blocs "details" des films
        movies = soup.find_all("div", class_="details")
        
        for movie in movies:
            # Extraire le titre du film
            title_tag = movie.find("a", href=True)
            title = title_tag.text.strip() if title_tag else "Titre non trouvé"
            
            # Extraire la note depuis l'attribut data-percent
            user_score_chart = movie.find("div", class_="user_score_chart")
            rating = user_score_chart.get("data-percent") if user_score_chart else "Note non trouvée"
            
            # Étape 2 : Rechercher le film par son nom
            movies = search_movie_by_name(title)

            # Étape 3 : Récupérer les genres
            genres_map = get_genres_map()

            # Étape 4 : Afficher les résultats
            print(f"\nRésultats pour le film '{title}' :\n")
            for movie in movies:
                title = movie["title"]
                release_date = movie.get("release_date", "Date inconnue")
                rating = movie.get("vote_average", "Note inconnue")
                genre_ids = movie.get("genre_ids", [])
                genres = [genres_map[g_id] for g_id in genre_ids if g_id in genres_map]
                
                if isinstance(rating, str):
                    continue
                rating = int(rating)
                if(rating>=5):
                    genres_movies.append(genres)
        return transform_genres(genres_list=genres_movies)

