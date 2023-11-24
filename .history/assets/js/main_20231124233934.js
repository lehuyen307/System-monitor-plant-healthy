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
    // const serviceLinkItem = document.querySelectorAll(".service__link");
    // const serviceHeading = document.querySelector(".service__heading.title");
    // [...serviceLinkItem].forEach((item) =>
    //     item.addEventListener("click", function (e) {
    //         if (e.target.dataset.tree == "1") {
    //             fetch("template.html")
    //                 .then((response) => response.text())
    //                 .then((data) => {
    //                     document.querySelector(".sick").innerHTML = data;
    //                     serviceHeading.textContent = "Nội dung trang 1";
    //                     window.location.href = "template.html";
    //                 });
    //         } else if (e.target.dataset.tree == "2") {
    //             console.log("hehee");
    //         }
    //     })
    // );

    // object
    let listSickTree = [
        {
            "name": "Bệnh bạc lá muộn ở cây cà chua",
            "text-1": "Bệnh bạc lá muộn (hay bệnh sương mai) là một bệnh nghiêm trọng gây hại cho cây cà chua. Bệnh do nấm Phytophthora infestans gây ra, nấm này phát triển mạnh trong điều kiện nhiệt độ thấp và ẩm ướt.",
            "text-2": "Trên lá: Vết bệnh ban đầu có màu xanh đậm như úng nước, sau đó chuyển sang nâu đen, vết bệnh lớn dần, nếu trời ẩm trên bề mặt lớp bệnh có lớp tơ màu trắng bao phủ, bệnh nặng sẽ làm thối nhũn lá, nếu thời tiết khô, vết trên lá cũng khô dòn dễ vỡ Trên thân cành: Bị bệnh từng đoạn dài, vỏ và ruột thân lúc đầu màu nâu hoặc thâm đen, sau đó thối ướt màu nâu đen.Trên quả: Vết bệnh ban đầu có màu xanh nhạt, sau đó chuyển sang nâu đen, vết bệnh lan rộng ra xung quanh, làm cho quả bị thối nhũn",
            "text-3": "Sử dụng giống cà chua kháng bệnh. Luân canh cây trồng, không trồng cà chua liên tục nhiều vụ trên cùng một ruộng. Tưới nước hợp lý, tránh tưới nước quá nhiều. Thu gom và tiêu hủy tàn dư cây trồng sau thu hoạch. Phun thuốc trừ bệnh theo khuyến cáo của nhà sản xuất. Phun thuốc trừ bệnh bạc lá muộn cần thực hiện khi bệnh chớm xuất hiện hoặc phun phòng khi thời tiết thích hợp cho bệnh phát triển. Nên phun thuốc vào lúc chiều mát để thuốc có thời gian bám dính tốt hơn",
            "value": "number-1"
        },
        {
            "name": "Bệnh bạc lá sớm ở cây cà chua",
            "text-1": "Bệnh bạc lá sớm do nấm Alternaria solani gây ra. Nấm bệnh tồn tại trong đất, trên tàn dư thực vật và trong hạt giống. Nấm bệnh phát triển mạnh trong điều kiện thời tiết ẩm ướt, nhiệt độ từ 20 đến 25 độ C.",
            "text-2": "Trên lá: Ban đầu là những đốm nhỏ màu xanh đậm, sau đó chuyển sang màu nâu đen, có hình tròn hoặc bất định. Vết bệnh thường tập trung ở lá già, sau đó lan dần lên các lá non. Khi bệnh nặng, lá bị rụng sớm, ảnh hưởng đến năng suất cà chua Trên thân cành: Vết bệnh có hình tròn hoặc elip, màu nâu đen, thường xuất hiện ở phần tiếp giáp giữa thân và lá. Khi bệnh nặng, thân bị thối nhũn, cây cà chua dễ bị đổ ngã. Trên quả: Vết bệnh có hình tròn hoặc bầu dục, màu nâu đen, thường xuất hiện ở phần cuống quả. Khi bệnh nặng, quả bị thối, không thể sử dụng được.",
            "text-3": "Sử dụng giống cà chua kháng bệnh. Luân canh cây trồng, không trồng cà chua trên đất đã trồng cà chua liên tiếp trong nhiều vụ. Vệ sinh đồng ruộng, thu gom và tiêu hủy tàn dư thực vật. Tưới nước vừa đủ, tránh tưới nước quá nhiều. Phun thuốc phòng trừ bệnh định kỳ, khi cây cà chua bắt đầu ra hoa.: Carbendazim, Mancozeb, Chlorothalonil, Propiconazole, Tebuconazole, Fluazinam",
            "value": "number-2"
        }
    ]
    console.log(listSickTree[1])

    function displayInfoSickTree(numberSick) {
        let sickNumber = listSickTree[numberSick];
        document.querySelector(".service__heading.title").textContent = ;

    }
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
