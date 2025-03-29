document.addEventListener("DOMContentLoaded", function () {
    console.log("Script loaded successfully!");

    // Image fullscreen functionality
    document.querySelectorAll("img").forEach(img => {
        img.addEventListener("click", function () {
            openFullScreen(this);
        });
    });

    function openFullScreen(img) {
        let fullScreenDiv = document.createElement("div");
        fullScreenDiv.classList.add("fullscreen-overlay");
        fullScreenDiv.innerHTML = `<img src="${img.src}" class="fullscreen-img">
                                   <span class="close-btn">×</span>`;
        document.body.appendChild(fullScreenDiv);

        fullScreenDiv.addEventListener("click", function (event) {
            if (event.target.classList.contains("close-btn") || event.target === fullScreenDiv) {
                fullScreenDiv.remove();
            }
        });
    }

    // Toggle login/signup forms
    const toggleSwitch = document.getElementById("toggle-switch");
    const loginForm = document.getElementById("login-form");
    const signupForm = document.getElementById("signup-form");

    if (toggleSwitch) {
        toggleSwitch.addEventListener("change", function () {
            console.log("Login/Signup toggle:", this.checked);
            if (this.checked) {
                loginForm.style.display = "none";
                signupForm.style.display = "block";
            } else {
                loginForm.style.display = "block";
                signupForm.style.display = "none";
            }
        });
    }

    // Theme toggle
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;

    // Set default theme to dark
    body.classList.add("dark-mode");

    if (themeToggle) {
        themeToggle.addEventListener("change", function () {
            console.log("Theme toggle:", this.checked);
            if (this.checked) {
                body.classList.remove("dark-mode");
                body.classList.add("light-mode");
            } else {
                body.classList.remove("light-mode");
                body.classList.add("dark-mode");
            }
        });
    }
});

