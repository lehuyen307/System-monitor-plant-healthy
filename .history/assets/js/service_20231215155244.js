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
