/**
 * Main JavaScript for the Medicinal Leaf Classifier prediction page.
 *
 * Handles image upload via AJAX and displays prediction results
 * with animated plant information sections.
 */

document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("upload-form");
    if (!uploadForm) return;

    uploadForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const resultElement = document.getElementById("prediction-result");
        resultElement.textContent = "Loading...";

        const formData = new FormData(this);

        fetch("/upload", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "success") {
                    // Hide instructions and main text
                    const instructions = document.getElementById("upload-instructions");
                    const mainContent = document.getElementById("main");
                    if (instructions) instructions.style.display = "none";
                    if (mainContent) mainContent.style.display = "none";

                    // Display uploaded image preview
                    const imageContainer = document.getElementById("image-preview-container");
                    imageContainer.innerHTML = `
                        <img class="img-fluid" src="${data.image_url}" alt="Uploaded leaf"
                             style="padding: 5px; border-radius: 10%; max-height: 190px; max-width: 190px;">
                        <br>
                    `;

                    // Display prediction result
                    resultElement.textContent = data.result;

                    // Show the matching plant info section
                    showPlantInfo(data.result);
                } else {
                    alert(data.error || "An error occurred");
                }
            })
            .catch((error) => {
                console.error("Upload error:", error);
                alert("Error uploading image. Please try again.");
            });
    });
});


/**
 * Show the plant information section matching the prediction result.
 *
 * Hides all plant sections first, then slides down the matching one.
 *
 * @param {string} plantName - The predicted plant class name.
 */
function showPlantInfo(plantName) {
    const allPlants = document.querySelectorAll(".plants");

    // Hide all sections
    allPlants.forEach((section) => {
        section.style.display = "none";
    });

    // Show the matching section
    const targetSection = document.querySelector(`.${plantName}`);
    if (targetSection) {
        targetSection.style.display = "block";

        // If jQuery is available, use slide animation
        if (typeof $ !== "undefined") {
            $(`.${plantName}`).hide().slideDown(2000);
        }
    }
}
