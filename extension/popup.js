document.addEventListener("DOMContentLoaded", () => {
    // Check if chrome.storage is available
    if (chrome.storage && chrome.storage.local) {
      // Load saved state
      chrome.storage.local.get(["activeSection", "tmdbUsername", "selectedGenres"], (result) => {
        if (chrome.runtime.lastError) {
          console.error("Error loading saved state:", chrome.runtime.lastError)
        } else {
          if (result.activeSection) {
            showSection(result.activeSection)
          }
          if (result.tmdbUsername) {
            document.getElementById("tmdb-username-input").value = result.tmdbUsername
          }
          if (result.selectedGenres) {
            result.selectedGenres.forEach((genre) => {
              const checkbox = document.querySelector(`input[value="${genre}"]`)
              if (checkbox) {
                checkbox.checked = true
              }
            })
          }
        }
      })
    } else {
      console.warn("chrome.storage is not available. State will not be saved.")
    }
  
    // Prevent the popup from closing when clicking outside
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "keepOpen" })
    })
  
    // Listen for messages from the content script
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "closePopup") {
        window.close()
      }
    })
    const mainMenu = document.getElementById("main-menu")
    const tmdbAccountSection = document.getElementById("tmdb-account-section")
    const categoriesSection = document.getElementById("categories-section")
    const backButtons = document.querySelectorAll(".back-button")
  
    // Main Menu Navigation
    document.getElementById("go-to-tmdb-account").addEventListener("click", () => {
      showSection("tmdb-account-section")
    })
  
    document.getElementById("go-to-categories").addEventListener("click", () => {
      showSection("categories-section")
    })
  
    // Back Buttons
    backButtons.forEach((button) => {
      button.addEventListener("click", () => {
        showSection("main-menu")
      })
    })
  
    // TMDB Account Recommendation
    document.getElementById("send-recommendation_by_account").addEventListener("click", () => {
      const usernameInput = document.getElementById("tmdb-username-input")
      const recommendationsContainer = document.getElementById("tmdb-recommendations")
      const username = usernameInput.value.trim()
  
      if (username) {
        fetch("http://localhost:5000/recommend", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ username: username }),
        })
          .then((response) => response.json())
          .then((data) => {
            recommendationsContainer.innerHTML = ""
            if (data.recommendations && data.recommendations.length > 0) {
              data.recommendations.forEach((recommendation) => {
                const recommendationElement = document.createElement("a")
                recommendationElement.href = recommendation.link
                recommendationElement.textContent = recommendation.title
                recommendationElement.target = "_blank"
                recommendationsContainer.appendChild(recommendationElement)
              })
            } else {
              recommendationsContainer.innerHTML = "<p>Aucune recommandation trouvée.</p>"
            }
           
            saveState()
          })
          .catch((error) => {
            console.error("Error sending recommendation:", error)
            recommendationsContainer.innerHTML = "<p>Erreur lors de la récupération des recommandations.</p>"
          })
      }
    })
  
    // Category Recommendation
    document.getElementById("send-recommendation_by_categorie").addEventListener("click", () => {
      const recommendationsContainer = document.getElementById("category-recommendations")
  
      // Get selected genres as a comma-separated string
      const selectedGenres = Array.from(document.querySelectorAll('input[name="genre"]:checked'))
        .map((checkbox) => checkbox.value)
        .join("|")
      if (selectedGenres) {
        fetch("http://localhost:5000/categorie", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ recommendation: selectedGenres }),
        })
          .then((response) => response.json())
          .then((data) => {
            recommendationsContainer.innerHTML = ""
            if (data.recommendations && data.recommendations.length > 0) {
              data.recommendations.forEach((recommendation) => {
                const recommendationElement = document.createElement("a")
                recommendationElement.href = recommendation.link
                recommendationElement.textContent = recommendation.title
                recommendationElement.target = "_blank"
                recommendationsContainer.appendChild(recommendationElement)
              })
            } else {
              recommendationsContainer.innerHTML = "<p>Aucune recommandation trouvée.</p>"
            }
            saveState()
          })
          .catch((error) => {
            console.error("Error sending recommendation:", error)
            recommendationsContainer.innerHTML = "<p>Erreur lors de la récupération des recommandations.</p>"
          })
      }
    })
  
    // Add event listeners for state changes - moved here to be consistent with the updated structure.
    document.getElementById("tmdb-username-input").addEventListener("change", saveState)
    document.querySelectorAll('input[name="genre"]').forEach((checkbox) => {
      checkbox.addEventListener("change", saveState)
    })
  })
  
  function saveState() {
    if (chrome.storage && chrome.storage.local) {
        const activeSection = document.querySelector(".menu-section.active")?.id;
        const tmdbUsername = document.getElementById("tmdb-username-input")?.value;
        const selectedGenres = Array.from(document.querySelectorAll('input[name="genre"]:checked')).map((cb) => cb.value);
        const categoryRecommendations = document.getElementById("category-recommendations")?.innerHTML || "";
        const tmdbRecommendations = document.getElementById("tmdb-recommendations")?.innerHTML || "";

        chrome.storage.local.set(
            {
                activeSection: activeSection,
                tmdbUsername: tmdbUsername,
                selectedGenres: selectedGenres,
                categoryRecommendations: categoryRecommendations,
                tmdbRecommendations: tmdbRecommendations,
            },
            () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving state:", chrome.runtime.lastError);
                }
            }
        );
    }
}

  
  function showSection(sectionId) {
    document.querySelectorAll(".menu-section").forEach((section) => {
      section.classList.remove("active")
    })
    const sectionToShow = document.getElementById(sectionId)
    if (sectionToShow) {
      sectionToShow.classList.add("active")
      saveState()
    } else {
      console.error(`Section with id ${sectionId} not found`)
    }
  }
  
  