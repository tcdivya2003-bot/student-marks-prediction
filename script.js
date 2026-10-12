// script.js - Student Marks Prediction Based on Study Hours
// Small client-side enhancement: basic form validation feedback.

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");

    if (form) {
        form.addEventListener("submit", function (event) {
            const input = document.getElementById("study_hours");
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);

            if (input.value.trim() === "" || isNaN(value) || value < min || value > max) {
                input.classList.add("is-invalid");
                event.preventDefault();
            } else {
                input.classList.remove("is-invalid");
                input.classList.add("is-valid");
            }
        });
    }
});
