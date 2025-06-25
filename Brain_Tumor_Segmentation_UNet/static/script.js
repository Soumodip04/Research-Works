document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('file');
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/classify', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (result.error) {
        alert(result.error);
    } else {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `
            <h2>Prediction:</h2>
            <p>Class: ${result.class}</p>
            <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
        `;
    }
});
