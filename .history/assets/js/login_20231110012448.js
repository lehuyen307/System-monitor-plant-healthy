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

// function showSuccess(input) {
//     const small = input.nextElementSibling;
//     small.textContent = "";
// }

// function showErr(input, message) {
//     const small = input.nextElementSibling;
//     small.textContent = message;
// }

// function getFieldName(input) {
//     return input.id.charAt(0).toUpperCase() + input.id.slice(1);
//     // slice thì tạo ra mảng mới và sao chép ra từ vị trí index truyền vào
//     // splice thì cắt hoặc xóa từ mảng gốc
// }

// function checkRequired(listInput) {
//     let check = false;
//     listInput.forEach((input) => {
//         if (input.value.trim() === "") {
//             // nếu giá trị rỗng
//             showErr(input, `${getFieldName(input)} is required`);
//             check = true;
//         } else {
//             showSuccess(input);
//         }
//     });
// }

// function checkEmail(input) {
//     const regex =
//         /^(([^<>()[\]\.,;:\s@\"]+(\.[^<>()[\]\.,;:\s@\"]+)*)|(\".+\"))@(([^<>()[\]\.,;:\s@\"]+\.)+[^<>()[\]\.,;:\s@\"]{2,})$/i;
//     if (regex.test(input.value)) {
//         showSuccess(input);
//     } else {
//         showErr(input, `${getFieldName(input)} is not valid`);
//     }
// }

// function checkLength(input, min, max) {
//     if (input.value.length < min) {
//         showErr(input, `${getFieldName(input)} có tối thiểu ${min} kí tự`);
//     } else if (input.value.length > max) {
//         showErr(input, `${getFieldName(input)} có tối đa ${max} kí tự`);
//     } else {
//         showSuccess(input);
//     }
// }

// function checkPassWord(input) {
//     const regexPass = /^(?=.*[A-Z])(?=.*[0-9])(?=.*[@#$%^&+=!])(?=.{8,})/;
//     if (regexPass.test(input.value)) {
//         showSuccess(input);
//     } else {
//         showErr(
//             input,
//             `${getFieldName(
//                 input
//             )} is least 8 characters, 1 uppercase letter, 1 number & 1 symbol`
//         );
//     }
// }

// function comparePass(pass1, pass2) {
//     if (pass1.value !== pass2.value) {
//         showErr(pass1, `${getFieldName(pass1)} password incorrect`);
//     } else {
//         showSuccess(pass1);
//     }
// }

function showErrorMessage() {
    const errText = document.querySelector(".textErr");
    errText.style.display = "block"
}

function handleInput() {
    
}

// API
const endPoint = "https://dm92v6-8080.csb.app/loginSystemTree";

// add api
async function addInfoSignUp({
    nameSignUp,
    emailSignUp,
    passwordSignUp,
    passwordSignUp2,
}) {
    const response = await fetch(endPoint, {
        method: "POST",
        body: JSON.stringify({
            nameSignUp,
            emailSignUp,
            passwordSignUp,
            passwordSignUp2,
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8",
        },
    });
}
// xử lý đăng ký (sign-up)
const formSignUp = document.querySelector(".form-SignUp");
const popup = document.querySelector(".pop-up");
formSignUp.addEventListener("submit", async function (e) {
    e.preventDefault();
    // let check = false; // thêm biến check để kiểm tra lỗi
    // if (
    //     !checkRequired([
    //         nameSignUp,
    //         emailSignUp,
    //         passwordSignUp,
    //         passwordSignUp2,
    //     ])
    // ) {
    //     checkLength(nameSignUp, 6, 15);
    //     checkEmail(emailSignUp);
    //     checkPassWord(passwordSignUp);
    //     checkPassWord(passwordSignUp2);
    //     comparePass(passwordSignUp, passwordSignUp2);
    //     check = true; // nếu có lỗi thì đổi thành true
    // }

    // if (check) {
    //     alert("Nhập chưa đúng"); // Thông báo lỗi
    //     return; // dừng
    // }

    // Tiếp tục xử lý khi không có lỗi
    const login = {
        nameSignUp: this.elements["nameSignUp"].value,
        emailSignUp: this.elements["emailSignUp"].value,
        passwordSignUp: this.elements["passwordSignUp"].value,
        passwordSignUp2: this.elements["passwordSignUp2"].value,
    };
    this.reset();
    await addInfoSignUp(login);
    popup.classList.remove("hidden");
    popup.classList.add("visible");
});

document.body.addEventListener("click", function (e) {
    if (e.target.matches(".btn-Popup")) {
        const move = e.target.parentNode.parentNode;
        move.classList.add("hidden");
        move.classList.remove("visible");
    }
});

// xử lý đăng nhập (sign-in)
async function fetchData() {
    const response = await fetch(endPoint);
    const data = await response.json();
    return data;
}

const formSignIn = document.querySelector(".form-login");
formSignIn.addEventListener("submit", async function (e) {
    e.preventDefault();
    const data = await fetchData();
    const email = this.elements["email"].value;
    const pass = this.elements["password"].value;

    const found = data.find(
        (item) => item.emailSignUp == email && item.passwordSignUp == pass
    ); // nếu tìm thấy thì thành công và ngược lại
    if (found) {
        alert("thanh cong");
    } else {
        alert("that bai");
    }
});
