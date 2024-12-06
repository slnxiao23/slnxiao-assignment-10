document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const resultsSection = document.getElementById('resultsSection');
    const resultsList = document.getElementById('resultsList');
    const usePcaCheckbox = document.getElementById('use_pca');
    const pcaField = document.getElementById('pcaField');

    // Toggle the PCA field based on the "Use PCA Embeddings" checkbox
    usePcaCheckbox.addEventListener('change', () => {
        if (usePcaCheckbox.checked) {
            pcaField.style.display = 'block'; // Show the PCA field
        } else {
            pcaField.style.display = 'none'; // Hide the PCA field
        }
    });

    // Handle form submission with AJAX
    searchForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        // Prepare the form data
        const formData = new FormData(searchForm);

        try {
            // Send an AJAX request to the server
            const response = await fetch('/search', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to fetch results');
            }

            // Parse the JSON response
            const data = await response.json();

            // Update the results section
            resultsList.innerHTML = ''; // Clear previous results
            if (data.results && data.results.length > 0) {
                data.results.forEach((result) => {
                    const listItem = document.createElement('li');
                    listItem.classList.add('result-item');

                    const img = document.createElement('img');
                    img.src = `/media/${result.file_name}`;
                    img.alt = result.file_name;
                    img.classList.add('result-image');

                    const similarity = document.createElement('p');
                    similarity.textContent = `Similarity: ${result.similarity.toFixed(3)}`;

                    listItem.appendChild(img);
                    listItem.appendChild(similarity);
                    resultsList.appendChild(listItem);
                });
                resultsSection.style.display = 'block';
            } else {
                resultsSection.style.display = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your search.');
        }
    });
});
