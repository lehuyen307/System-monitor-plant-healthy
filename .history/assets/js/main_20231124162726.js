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

    // xử lý button ở gioi thieu
    // scrollLeft là thuộc tính left nó dựa trên thanh cuộn scroll (cuộn đi)
    // offsetLeft là thuộc tính left dựa trên viewport (không cộng dồn được trên thanh scroll)
    const btnLeft = document.querySelector(".service__button.left");
    const btnRight = document.querySelector(".service__button.right");
    const serviceList = document.querySelector(".service__list");
    btnLeft.addEventListener("click", function () {
        const a = serviceList.scrollLeft;
        serviceList.scroll(a - 300, 0);
    });
    btnRight.addEventListener("click", function () {
        const a = serviceList.scrollLeft;
        serviceList.scroll(a + 300, 0);
    });

    // xử lý button rating
    const btnLeftRate = document.querySelector(".service__btn.left1");
    const btnRightRate = document.querySelector(".service__btn.right1");
    const rateList = document.querySelector(".service__rateList");
    btnLeftRate.addEventListener("click", function (e) {
        const b = rateList.scrollLeft;
        rateList.scroll(b - 300, 0);
    });
    btnRightRate.addEventListener("click", function (e) {
        const b = rateList.scrollLeft;
        rateList.scroll(b + 300, 0);
    });

    // xử lý button chuyển trang
    const serviceLinkItem = document.querySelectorAll(".service__link");
    const serviceHeading = document.querySelector(".service__heading.title");
    [...serviceLinkItem].forEach(item => item.addEventListener("click", function(e) {
        if(e.target.dataset.tree == "1") {
            fetch("template.html").then(response =)
        } else if (e.target.dataset.tree == "2") {
            console.log("hehee");
        } 
    }))
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

if ("IntersectionObserver" in window) {
    const options = {
        threshold: 0.5,
    };
    let observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                // handle logic here
                entry.target.src = entry.target.dataset.src;
                // observer.unobserve(entry.target)
                // Sau khi xử lý logic, chúng ta gọi phương thức unobserve() trên đối tượng observer để ngừng quan sát phần tử đó.
                observer.unobserve(entry.target);
            }
        });
        // console.log(entries)
    }, options);

    const images = document.querySelectorAll(".img");
    images.forEach((item) => observer.observe(item));
}
