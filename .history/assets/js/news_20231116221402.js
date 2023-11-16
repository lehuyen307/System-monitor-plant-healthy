window.addEventListener("load", function() {
    const newsBox = document.querySelector(".news__box")
    newsBox.addEventListener("click", function() {
        newsBox.classList.toggle("visible")
        newsBox.classList.toggle("hidden")
    })
})