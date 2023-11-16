window.addEventListener("load", function() {
    const newsBox = document.querySelector(".news__box")
    const newsLanguage = document.querySelector(".news__language")
    newsBox.addEventListener("click", function() {
        newsBox.classList.toggle("visible")
        newsBox.classList.toggle("hidden")
    })
})