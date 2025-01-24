import psutil
import time
from functools import wraps  # Correction ici
from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.scrapper import *
from backend.clip.clip import *

app = Flask(__name__)
CORS(app)

# Fonction pour surveiller la consommation électrique
def monitor_energy_during_request(func):
    @wraps(func)  # Assure la compatibilité avec Flask
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        start_memory = process.memory_info().rss  # Mémoire en octets
        start_time = time.time()
        
        # Appeler la fonction Flask originale
        result = func(*args, **kwargs)
        
        # Après l'exécution
        elapsed_time = time.time() - start_time
        end_cpu_times = process.cpu_times()
        end_memory = process.memory_info().rss  # Mémoire en octets
        
        # Calcul des ressources consommées
        cpu_time = end_cpu_times.user - start_cpu_times.user
        memory_usage_mb = (end_memory - start_memory) / 1024**2  # En Mo
        
        # Estimation énergétique (approximative)
        tdp_cpu = 28  # Exemple TDP CPU (à ajuster selon ton processeur)
        energy_cpu = tdp_cpu * (cpu_time / 3600)  # Wh
        print(f"Temps CPU : {cpu_time:.2f} s, Mémoire : {memory_usage_mb:.2f} Mo, Énergie CPU : {energy_cpu:.2f} Wh")
        
        return result
    
    return wrapper

@app.route('/recommend', methods=['POST'])
@monitor_energy_during_request
def recommend():
    recommendation = request.get_json()['username']
    genres = get_profil_moovie(recommendation)
    print(genres)
    recommendations_df = get_recomandation(genres)

    recommendations = [
        {"title": row['title'], "link": f"https://www.google.com/search?q={row['title']}"} 
        for _, row in recommendations_df.iterrows()
    ]
    print(jsonify({"recommendations": recommendations}))
    return jsonify({"recommendations": recommendations})

@app.route('/categorie', methods=['POST'])
@monitor_energy_during_request
def categorie():
    genre = request.get_json()['recommendation']
    recommendations_df = get_recomandation(genre)

    recommendations = [
        {"title": row['title'], "link": f"https://www.google.com/search?q={row['title']}"} 
        for _, row in recommendations_df.iterrows()
    ]
    print(recommendations)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)

    #OrthrosOS