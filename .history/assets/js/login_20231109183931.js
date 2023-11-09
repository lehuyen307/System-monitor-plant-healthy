window.addEventListener("load", function () {
    // xử lý giao diện login
    // const signUp = document.querySelector(".signUp");
    const form1 = document.querySelector(".form.form-1");
    const form2 = document.querySelector(".form.form-2");
    // signUp.addEventListener("click", function (e) {
    //     console.log("kaa");
    // });

    const signUp = document.querySelector(".signUp");
    signUp.addEventListener("click", function (e) {
        form1.classList.add("visible");
        form1.classList.remove("hidden");
        form2.classList.remove("visible");
        form2.classList.add("hidden");
    });
    const signIn = document.querySelector(".signIn");
    signIn.addEventListener("click", function (e) {
        form1.classList.remove("visible");
        form1.classList.add("hidden");
        form2.classList.add("visible");
        form2.classList.remove("hidden");
    });

    // exit login
    const buttonBack = document.querySelector(".button-back")
    buttonBack.addEventListener("click", function() {
        window.location = "index.html"
    })

    // change icon pass
    const iconEye = document.querySelectorAll("")
});
