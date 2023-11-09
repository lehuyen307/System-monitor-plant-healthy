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
    const buttonBack = document.querySelector(".button-back");
    buttonBack.addEventListener("click", function () {
        window.location = "index.html";
    });

    // change icon pass
    const iconEye1 = document.querySelector(".fa-solid.icon-1");
    iconEye1.addEventListener("click", function(e) {
        iconEye1.classList.toggle("fa-eye-slash")
        iconEye1.classList.toggle("fa-eye")
        if(passwordSignUp.type == "password" ) {
            passwordSignUp.type = "text"
        } else {
            passwordSignUp.type = "password"
        }
    })
    const iconEye2 = document.querySelector(".fa-solid.icon-2");
    iconEye2.addEventListener("click", function(e) {
        iconEye2.classList.toggle("fa-eye-slash")
        iconEye2.classList.toggle("fa-eye")
        if(passwordSignUp2.type == "password" ) {
            passwordSignUp2.type = "text"
        } else {
            passwordSignUp2.type = "password"
        }
    })

    // xử lý form signUp (đăng ký)
    const formSignUp = document.querySelector(".form-SignUp")
    formSignUp.addEventListener("submit", function(e) {
        e.preventDefault()
        const a = this.elements["nameSignUp"].value
        const b = this.elements["emailSignUp"].value
        const c = this.elements["passwordSignUp"].value
        const d = this.elements["passwordSignUp2"].value
        console.log(a,b,c,d)
    })

    const nameSignUp = document.getElementById("nameSignUp")
    console.log(nameSignUp)
});
