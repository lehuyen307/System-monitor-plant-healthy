window.addEventListener("load", function () {
    const newsBox = document.querySelector(".news__box");
    const newsLanguage = document.querySelector(".news__language");
    const language = document.querySelectorAll(".language");
    const textMain = do
    newsBox.addEventListener("click", function () {
        newsLanguage.classList.toggle("visible");
        newsLanguage.classList.toggle("hidden");
    });

    // [...name] // lấy toàn bộ // mảng
    [...language].forEach((item) =>
        item.addEventListener("click", function (e) {
            const textName = e.target.textContent;
            console.log(textName);
        })
    );
});
