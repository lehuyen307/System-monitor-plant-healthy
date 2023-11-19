window.addEventListener("load", function (e) {
    function debounceFn(func, wait, immediate) {
        let timeout;
        return function () {
            let context = this,
                args = arguments;
            let later = function () {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            let callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    }

    // xử lý header scroll
    const header = document.querySelector(".header");
    const heightHeader = header && header.offsetHeight;
    window.addEventListener(
        "scroll",
        debounceFn(function (e) {
            const pageY = window.scrollY;
            if (pageY > heightHeader) {
                header.classList.add("is-fixed");
                document.body.style.paddingTop = `${heightHeader}px`;
            } else {
                header.classList.remove("is-fixed");
                document.body.style.paddingTop = `0px`;
            }
        }, 100)
    );

    // xử lý hover nav
    const headerNavLink = document.querySelectorAll(".header-list__link");
    const headerList = document.querySelector(".header-list");
    const line = document.createElement("div");
    line.className = "line";
    document.body.appendChild(line);
    [...headerNavLink].forEach((item) => {
        item.addEventListener("mouseenter", function (e) {
            const cords = e.target.getBoundingClientRect(); // lấy tọa độ
            const { width, height, top, left } = cords;
            line.style.width = `${width}px`;
            line.style.left = `${left}px`;
            line.style.top = `${top + height - 2}px`;
        });
    });
    headerList.addEventListener("mouseleave", function (e) {
        line.style.width = 0;
    });

    const headerLinkContact = document.querySelector(
        ".header-list__link.contact"
    );
    headerLinkContact.addEventListener("click", function (e) {
        const page = window.scrollTo(0,1900);
    })
});

document.addEventListener("DOMContentLoaded", function () {
    const userName = sessionStorage.getItem("userName");
    if (userName) {
        // xử lý ở phần header // không liên quan các trang main
        const headerBtn = document.querySelector(".header__btn");
        headerBtn.style.display = "none";
        const headerIcon = document.querySelector(".header__icon");
        headerIcon.style.display = "none";
        const btnLogin = document.querySelector(".btn__login");
        btnLogin.style.display = "block";
        const userNameView = document.querySelector(".userName");
        userNameView.textContent = userName;

        const btnLogout = document.querySelector(".btn__logout");
        btnLogin.addEventListener("click", function (e) {
            btnLogout.classList.toggle("visible");
            btnLogout.classList.toggle("hidden");
        });

        btnLogout.addEventListener("click", function (e) {
            headerBtn.style.display = "block";
            headerIcon.style.display = "block";
            btnLogin.style.display = "none";
            sessionStorage.removeItem("userName");
            btnLogout.classList.add("hidden");
            btnLogout.classList.remove("visible");
        });
    }
});
