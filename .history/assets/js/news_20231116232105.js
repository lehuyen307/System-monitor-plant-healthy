window.addEventListener("load", function () {
    const newsBox = document.querySelector(".news__box");
    const newsLanguage = document.querySelector(".news__language");
    const language = document.querySelector(".language");
    const textMain = document.querySelector(".news__box .text");

    newsBox.addEventListener("click", function () {
        newsLanguage.classList.toggle("visible");
        newsLanguage.classList.toggle("hidden");
    });

    // [...name] // lấy toàn bộ // mảng
    language.addEventListener("click", function (e) {
        const textName = e.target.textContent;
        language.textContent = textMain.textContent;
        textMain.textContent = textName;
        newsLanguage.classList.remove("visible");
        newsLanguage.classList.add("hidden");
        const text = document.querySelector(".special")
        translate(text, { from: "vi", to: "en" })
            .then((res) => {
                document.querySelector(".special").innerText = res.text;
            })
            .catch((err) => {
                console.error(err);
            });
    });
});
