from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.scrapper import *
from backend.clip.clip import *
app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    recommendation = request.get_json()['username']
    genres=get_profil_moovie(recommendation)
    
    recommendations_df=get_recomandation(genres)

    recommendations = [
        {"title": row['title'], "link": f"https://www.imdb.com/title/"} 
        for _, row in recommendations_df.iterrows()
    ]
    print(jsonify({"recommendations":recommendations }))
    return jsonify({"recommendations":recommendations })

@app.route('/categorie', methods=['POST'])
def categorie():
    genre = request.get_json()['recommendation']
    
    recommendations_df=get_recomandation(genre)

    recommendations = [
        {"title": row['title'], "link": f"https://www.imdb.com/title/"} 
        for _, row in recommendations_df.iterrows()
    ]
    print(recommendations)
    return jsonify({"recommendations":recommendations })

if __name__ == '__main__':
    app.run(debug=True)
    #OrthrosOS