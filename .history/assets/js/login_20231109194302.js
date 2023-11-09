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
    iconEye1.addEventListener("click", function (e) {
        iconEye1.classList.toggle("fa-eye-slash");
        iconEye1.classList.toggle("fa-eye");
        if (passwordSignUp.type == "password") {
            passwordSignUp.type = "text";
        } else {
            passwordSignUp.type = "password";
        }
    });
    const iconEye2 = document.querySelector(".fa-solid.icon-2");
    iconEye2.addEventListener("click", function (e) {
        iconEye2.classList.toggle("fa-eye-slash");
        iconEye2.classList.toggle("fa-eye");
        if (passwordSignUp2.type == "password") {
            passwordSignUp2.type = "text";
        } else {
            passwordSignUp2.type = "password";
        }
    });

    const nameSignUp = document.getElementById("nameSignUp");
    const emailSignUp = document.getElementById("emailSignUp");
    const passwordSignUp = document.getElementById("passwordSignUp");
    const passwordSignUp2 = document.getElementById("passwordSignUp2");

    function showSuccess(input) {
        const small = input.nextElementSibling;
        small.textContent = "";
    }

    function showErr(input, message) {
        const small = input.nextElementSibling;
        small.textContent = message;
    }

    function getFieldName(input) {
        return input.id.charAt(0).toUpperCase() + input.id.slice(1);
        // slice thì tạo ra mảng mới và sao chép ra từ vị trí index truyền vào
        // splice thì cắt hoặc xóa từ mảng gốc
    }

    function checkRequired(listInput) {
        let check = false;
        listInput.forEach((input) => {
            if (input.value.trim() === "") {
                // nếu giá trị rỗng
                showErr(input, `${getFieldName(input)} is required`);
                check = true;
            } else {
                showSuccess(input);
            }
        });
    }

    function checkEmail(input) {
        const regex =
            /^(([^<>()[\]\.,;:\s@\"]+(\.[^<>()[\]\.,;:\s@\"]+)*)|(\".+\"))@(([^<>()[\]\.,;:\s@\"]+\.)+[^<>()[\]\.,;:\s@\"]{2,})$/i;
        if (regex.test(input.value)) {
            showSuccess(input);
        } else {
            showErr(input, `${getFieldName(input)} is not valid`);
        }
    }

    function checkLength(input, min, max) {
        if (input.value.length < min) {
            showErr(input, `${getFieldName(input)} có tối thiểu ${min} kí tự`);
        } else if (input.value.length > max) {
            showErr(input, `${getFieldName(input)} có tối đa ${max} kí tự`);
        } else {
            showSuccess(input);
        }
    }

    function checkPassWord(input) {
        const regexPass = /^(?=.*[A-Z])(?=.*[0-9])(?=.*[@#$%^&+=!])(?=.{8,})/;
        if (regexPass.test(input.value)) {
            showSuccess(input);
        } else {
            showErr(
                input,
                `${getFieldName(
                    input
                )} is least 8 characters, 1 uppercase letter, 1 number & 1 symbol`
            );
        }
    }

    function comparePass(pass1, pass2) {
        if(pass1.value === pass2.value) {
            
        }
    }

    // xử lý form signUp (đăng ký)
    const formSignUp = document.querySelector(".form-SignUp");
    formSignUp.addEventListener("submit", function (e) {
        e.preventDefault();
        if (
            !checkRequired([
                nameSignUp,
                emailSignUp,
                passwordSignUp,
                passwordSignUp2,
            ])
        ) {
            checkLength(nameSignUp,6, 15);
            checkEmail(emailSignUp);
            checkPassWord(passwordSignUp);
            checkPassWord(passwordSignUp2);
        }
    });
});
