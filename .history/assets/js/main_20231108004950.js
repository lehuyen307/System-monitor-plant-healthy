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

    const header = document.querySelector(".header");
    const heightHeader = header && header.offsetHeight;
    window.addEventListener(
        "scroll",
        debounceFn(function (e) {
            const pageY = window.scrollY;
            console.log(pageY)
            if(pageY > heightHeader) {
                header.classList.add("is-fixed");
                heightHeader.style.paddingTop = `${heightHeader}`
            } else {
                header.classList.remove("is-fixed")
                heightHeader.style.paddingTop = 
            }
        }, 100)
    );
});
