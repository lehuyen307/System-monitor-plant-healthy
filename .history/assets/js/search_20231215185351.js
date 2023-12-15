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
// dùng try catch thì async await đi cùng và không dùng then()

const url1 = "/listTinTuc";
const fetchSearch1 = async (link = url1) => {
  try {
    const res = await http.get(link);
    const data = res.data;
    headerListTreeSick.innerHTML = "";
    if (data.length > 0 && Array.isArray(data)) {
      [...data].forEach((item) => createItemSick(item));
    }
  } catch (err) {
    console.log(err);
  }
};
fetchSearch1();

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

const url2 = "/listSick"
const fetchSearch2 = (link = url2) => {
  http.get(link).then(res => {
    
  })
}
fetchSearch2()


const headerInput = document.querySelector(".header__input");
headerInput.addEventListener("keydown", function (e) {
  const searchItem = e.target.value;
  console.log(searchItem);
  const path = searchItem !== "" ? `${url1}?nameItem_like=${searchItem}` : url1;
  fetchSearch1(path);
});

