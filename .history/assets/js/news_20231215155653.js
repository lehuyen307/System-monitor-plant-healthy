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