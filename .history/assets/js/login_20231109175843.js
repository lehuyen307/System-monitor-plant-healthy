window.addEventListener("load", function () {
    // xử lý giao diện login
    const signUp = document.querySelector(".signUp");
    const form1 = document.querySelector(".form.form-1")
    const form2 = document.querySelector(".form.form-2")
    signUp.addEventListener("click", function (e) {
        console.log("kaa");
    }); 

    const signIn = document.querySelector(".signIn");
    signIn.addEventListener("click", function (e) {
        form1.classList.add("visible")
        form1.classList.add("hiddee")
    }); 
});
