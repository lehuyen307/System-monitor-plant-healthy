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
