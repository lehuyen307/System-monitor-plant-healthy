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

const http = axios.create({
  baseURL: `https://dm92v6-8080.csb.app`,
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

const fetchDataSearch_New = async () => {
  try {
    const response = await http.get("./listTinTuc", params: {
      
    })
  } catch (err) {
    console.log(err);
  }
};
const headerInput = document.querySelector(".header__input");
headerInput.addEventListener(
  "keydown",
  debounceFn((e) => {
    console.log(e.target.value);
    let path = endPointSearch_1;
    if (e.target.value !== "") {
      path = `${endPointSearch_1}?nameItem=${e.target.value}`;
    }
    fetchDataSearch_New(path);
  }, 500)
);
