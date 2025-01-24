# numerics


Pour appliquer la factorisation matricielle, nous devons structurer les données dans une matrice R où :

Chaque ligne représente un utilisateur.
Chaque colonne représente un élément (film ou série).
Les valeurs sont les votes donnés par les utilisateurs pour les films/séries.

Cependant, le dataset ne contient pas d'information explicite sur les utilisateurs.
Voici comment nous allons procéder :

Simuler des utilisateurs en générant des interactions aléatoires (votes) pour les éléments.
Construire une matrice R à partir des données simulées.
Appliquer la factorisation matricielle sur R.

Je vais commencer par simuler les données des utilisateurs.

# Extension de Recommandation de Films Chrome

## Prérequis
- Python 3.10.11
- Navigateur Google Chrome
- Environnement virtuel (recommandé)

## Étapes d'Installation

1. Cloner le Dépôt
```bash
git clone git@github.com:LIGHTER91/numerics.git
cd numerics
```

2. Créer un Environnement Virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

3. Installer les Dépendances
```bash
pip install -r Requierements.txt
```

4. Lancer le Serveur Backend
```bash
python app.py
```

## Configuration de l'Extension Chrome

1. Ouvrir Google Chrome
2. Aller sur `chrome://extensions/`
3. Activer le "Mode développeur" (bouton en haut à droite)
4. Cliquer sur "Charger l'extension non empaquetée"
5. Sélectionner le répertoire de l'extension Chrome "extension"

## Configuration
- S'assurer que `app.py` est en cours d'exécution avant d'utiliser l'extension
- Le serveur backend doit tourner sur `localhost:5000`