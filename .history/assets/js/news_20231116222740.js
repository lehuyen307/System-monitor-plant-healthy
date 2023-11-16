window.addEventListener("load", function() {
    const newsBox = document.querySelector(".news__box")
    const newsLanguage = document.querySelector(".news__language")
    const language = document.querySelectorAll(".language")
    newsBox.addEventListener("click", function() {
        newsLanguage.classList.toggle("visible")
        newsLanguage.classList.toggle("hidden")
    })
})