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