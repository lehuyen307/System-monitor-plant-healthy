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
  // const header = document.querySelector(".header");
  // const heightHeader = header && header.offsetHeight;
  // window.addEventListener(
  //   "scroll",
  //   debounceFn(function (e) {
  //     const pageY = window.scrollY;
  //     if (pageY > heightHeader) {
  //       header.classList.add("is-fixed");
  //       document.body.style.paddingTop = `${heightHeader}px`;
  //     } else {
  //       header.classList.remove("is-fixed");
  //       document.body.style.paddingTop = `0px`;
  //     }
  //   }, 100)
  // );

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
    serviceList.scroll(a - 500, 0);
  });
  btnRight.addEventListener("click", function () {
    const a = serviceList.scrollLeft;
    serviceList.scroll(a + 500, 0);
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

const newsList = document.querySelector(".news__list");
const createItem = (item) => {
  const template = `<div class="news__item">
  <a
    target="_blank"
    href="${item.linkItem}"
    class="news__link"
  >
    <img src="${item.imageItem}" alt="" class="news__img" />
    <span class="news__date">${item.dateItem} </span>
    <h4 class="news__heading">
      ${item.nameItem}
    </h4>
  </a>
</div>`;
  newsList.insertAdjacentHTML("beforeend", template);
};

// sử dụng fetch của trình duyệt hoặc thư viện axios
const imageLoader = document.querySelector(".image-loader");
let data = []; // mỗi lần fetch 1 trang là add vô data
let currentPage = 1;
const limit = 4;
const http = axios.create({
  baseURL: `https://dm92v6-8080.csb.app`,
});
function fetchData(page = 1) {
  http.get(`/listTinTuc?_limit=${limit}&_page=${currentPage}`).then((res) => {
    const newData = res.data;
    data = newData;
    imageLoader.style.display = "block";
    newsList.innerHTML = ""; // mỗi lần fetch lại trang thì cho newList rỗng để add data mới
    if (data.length > 0 && Array.isArray(data)) {
      [...data].forEach((item) => createItem(item));
      imageLoader.style.display = "none";
    }
  });
}
fetchData();

// cách 2
// const endPointListNews = `https://dm92v6-8080.csb.app/listTinTuc`;
// function fetchData(page = 1) {
//   axios({
//     baseURL: `${endPointListNews}`,
//     method: "get", // get dữ liệu về
//     url: `?_limit=${limit}&_page=${currentPage}`,
//   }).then((res) => {
//     const newData = res.data;
//     data = newData; // add vào mảng
//     newsList.innerHTML = "";
//     imageLoader.style.display = "block";
//     if (data.length > 0 && Array.isArray(data)) {
//       [...data].forEach((item) => createItem(item));
//       imageLoader.style.display = "none";
//     }
//   });
// }
// fetchData();

const endPointNew = `${http}`
const headerInput = document.querySelector(".header__input")
headerInput.addEventListener("keydown", (e) => {
  console.log(e.target.value);
})

const numberPage = document.querySelectorAll(".number");
[...numberPage].forEach((item, index) =>
  item.addEventListener("click", function (e) {
    const page = index + 1;
    currentPage = page;
    fetchData(page);
    window.scroll(0, 0);
  })
);

const iconSearch = document.querySelector(".header__search");
const headerSearchInfo = document.querySelector(".header__searchInfo");
const iconClose = document.querySelector(".header__close");
iconSearch.addEventListener("click", function (e) {
  headerSearchInfo.classList.toggle("visible");
  headerSearchInfo.classList.toggle("hidden");
});

document.body.addEventListener("click", (e) => {
  if (e.target.matches(".header__close")) {
    headerSearchInfo.classList.remove("visible");
    headerSearchInfo.classList.add("hidden");
  } else if (e.target.matches(".header__searchInfo")) {
    headerSearchInfo.classList.remove("visible");
    headerSearchInfo.classList.add("hidden");
  }
});

const headerListTreeSick = document.querySelector(".header__list");
const createItemSick = (item) => {
  const template = `<div class="header__item">
  <a
    target="_blank"
    href="${item.linkItem}"
    class="news__link"
  >
    <img src="${item.imageItem}" alt="" class="header__img" />
    <span class="header__date">${item.dateItem} </span>
    <h4 class="header__heading">
      ${item.nameItem}
    </h4>
  </a>
</div>`;

  headerListTreeSick.insertAdjacentHTML("beforeend", template);
};
const fetchDataSearch1 = () => {
  http.get("/listTinTuc").then((res) => {
    const data = res.data;
    if (data.length > 0 && Array.isArray(data)) {
      [...data].forEach((item) => createItemSick(item));
    }
  });
};
fetchDataSearch1();
const btnLeftHeader = document.querySelector(".btn__left");
const btnRightHeader = document.querySelector(".btn__right");
btnLeftHeader.addEventListener("click", function () {
  const a = headerListTreeSick.scrollLeft;
  headerListTreeSick.scroll(a - 300, 0);
});
btnRightHeader.addEventListener("click", function () {
  const a = headerListTreeSick.scrollLeft;
  headerListTreeSick.scroll(a + 300, 0);
});

const headerListTreeSick2 = document.querySelector(".header__list.list-2");
const createItemSick2 = (item) => {
  const template = `<div class="header__item">
  <a
    target="_blank"
    href="${item.linkItem}"
    class="news__link"
  >
    <img src="${item.imageItem}" alt="" class="header__img" />
    <h4 class="header__heading heading-2">
      ${item.nameItem}
    </h4>
  </a>
</div>`;
  headerListTreeSick2.insertAdjacentHTML("beforeend", template);
};
const fetchDataSearch2 = () => {
  http.get("/listSick").then((res) => {
    const data = res.data;
    if (data.length > 0 && Array.isArray(data)) {
      [...data].forEach((item) => createItemSick2(item));
    }
  });
};
fetchDataSearch2();

const btnLeftHeader2 = document.querySelector(".btn__left.left2");
const btnRightHeader2 = document.querySelector(".btn__right.right2");
btnLeftHeader2.addEventListener("click", function () {
  const a = headerListTreeSick2.scrollLeft;
  headerListTreeSick2.scroll(a - 300, 0);
});
btnRightHeader2.addEventListener("click", function () {
  const a = headerListTreeSick2.scrollLeft;
  headerListTreeSick2.scroll(a + 300, 0);
});
