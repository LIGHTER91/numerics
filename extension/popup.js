document.addEventListener('DOMContentLoaded', () => {
    const mainMenu = document.getElementById('main-menu');
    const tmdbAccountSection = document.getElementById('tmdb-account-section');
    const categoriesSection = document.getElementById('categories-section');
    const backButtons = document.querySelectorAll('.back-button');

    // Main Menu Navigation
    document.getElementById('go-to-tmdb-account').addEventListener('click', () => {
        mainMenu.classList.remove('active');
        tmdbAccountSection.classList.add('active');
    });

    document.getElementById('go-to-categories').addEventListener('click', () => {
        mainMenu.classList.remove('active');
        categoriesSection.classList.add('active');
    });

    // Back Buttons
    backButtons.forEach(button => {
        button.addEventListener('click', () => {
            tmdbAccountSection.classList.remove('active');
            categoriesSection.classList.remove('active');
            mainMenu.classList.add('active');
        });
    });

    // TMDB Account Recommendation
    document.getElementById('send-recommendation_by_account').addEventListener('click', () => {
        const usernameInput = document.getElementById('tmdb-username-input');
        const recommendationsContainer = document.getElementById('tmdb-recommendations');
        const username = usernameInput.value.trim();

        if (username) {
            fetch('http://localhost:5000/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username: username })
            })
            .then(response => response.json())
            .then(data => {
                recommendationsContainer.innerHTML = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach(recommendation => {
                        const recommendationElement = document.createElement('a');
                        recommendationElement.href = recommendation.link;
                        recommendationElement.textContent = recommendation.title;
                        recommendationElement.target = '_blank';
                        recommendationsContainer.appendChild(recommendationElement);
                    });
                } else {
                    recommendationsContainer.innerHTML = '<p>Aucune recommandation trouvée.</p>';
                }
                usernameInput.value = '';
            })
            .catch(error => {
                console.error('Error sending recommendation:', error);
                recommendationsContainer.innerHTML = '<p>Erreur lors de la récupération des recommandations.</p>';
            });
        }
    });

    // Category Recommendation
    document.getElementById('send-recommendation_by_categorie').addEventListener('click', () => {
        const recommendationsContainer = document.getElementById('category-recommendations');
        
        // Get selected genres as a comma-separated string
        const selectedGenres = Array.from(
            document.querySelectorAll('input[name="genre"]:checked')
        ).map(checkbox => checkbox.value).join('|');
        if (selectedGenres) {
            fetch('http://localhost:5000/categorie', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ recommendation: selectedGenres })
            })
            .then(response => response.json())
            .then(data => {
                recommendationsContainer.innerHTML = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach(recommendation => {
                        const recommendationElement = document.createElement('a');
                        recommendationElement.href = recommendation.link;
                        recommendationElement.textContent = recommendation.title;
                        recommendationElement.target = '_blank';
                        recommendationsContainer.appendChild(recommendationElement);
                    });
                } else {
                    recommendationsContainer.innerHTML = '<p>Aucune recommandation trouvée.</p>';
                }
             
            })
            .catch(error => {
                console.error('Error sending recommendation:', error);
                recommendationsContainer.innerHTML = '<p>Erreur lors de la récupération des recommandations.</p>';
            });
        }
    });
});