window.addEventListener("load", function () {
    const newsBox = document.querySelector(".news__box");
    const newsLanguage = document.querySelector(".news__language");
    const language = document.querySelectorAll(".language");
    const textMain = document.querySelector(".news__box .text");

    newsBox.addEventListener("click", function () {
        newsLanguage.classList.toggle("visible");
        newsLanguage.classList.toggle("hidden");
    });

    // [...name] // lấy toàn bộ // mảng
    [...language].forEach((item) =>
        item.addEventListener("click", function (e) {
            const textName = e.target.textContent;
            textMain.textContent = textName;
            newsLanguage.classList.remove("visible");
            newsLanguage.classList.add("hidden");

            if (e.target.dataset.lan == 1) {
                google.language = "vi";
                location.reload();
                console.log("1");
            } else if (e.target.dataset.lan == 2) {
                google.language = "en";
                location.reload();
                console.log("2");
            }
        })
    );
});

const translate = require("google-translate-api");

translate("Hello", { from: "en", to: "vi" })
    .then((res) => {
        console.log(res.text); // Kết quả dịch là "Xin chào"
    })
    .catch((err) => {
        console.error(err);
    });
