from flask import Flask, render_template, request, url_for
import os
import matplotlib.pyplot as plt
from utils.model_definitions import load_model
from utils.preprocessing import preprocess_image
from utils.evaluation import predict_classification, predict_segmentation

# Initialize Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/static/uploads/"
app.config["PREDICTION_FOLDER"] = "uploads/static/predictions/"
app.config["PLOT_FOLDER"] = "uploads/static/plots/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PREDICTION_FOLDER"], exist_ok=True)
os.makedirs(app.config["PLOT_FOLDER"], exist_ok=True)

# Load models
models = {
    "resnet18": load_model("models/resnet18_classification_model.pth", model_name="resnet18"),
    "resnet50": load_model("models/resnet50_classification_model.pth", model_name="resnet50"),
    "vgg16": load_model("models/vgg16_classification_model.pth", model_name="vgg16"),
    "unet": load_model("models/unet_segmentation_model.pth", model_name="unet"),
}

# Class names for classification models
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

def create_confidence_plot(confidences, class_names, plot_path):
    """Create a confidence plot and save it to the specified path."""
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, confidences, color=["blue", "orange", "green"])
    plt.xlabel("Classes")
    plt.ylabel("Confidence (%)")
    plt.title("Confidence Levels for Each Class")
    plt.ylim(0, 100)
    plt.savefig(plot_path)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        file = request.files.get("file")
        model_name = request.form.get("model")

        # Validate input
        if not file:
            return render_template("index.html", error="No file uploaded. Please upload an image.")
        if model_name not in models:
            return render_template("index.html", error="Invalid model selected. Please choose a valid model.")

        try:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Preprocess image
            input_tensor = preprocess_image(file_path, model_name)

            # Initialize prediction variables
            predicted_image_path = None
            plot_path = None

            # Perform prediction
            if model_name == "unet":
                result, confidence, predicted_image_path = predict_segmentation(
                    models[model_name], input_tensor, file_path, app.config["PREDICTION_FOLDER"]
                )
                result_text = result
            else:
                result_index, confidences = predict_classification(models[model_name], input_tensor)

                # Ensure confidences is iterable
                if not hasattr(confidences, "__iter__"):
                    raise ValueError("Confidences should be a list or array.")

                result_text = f"{CLASS_NAMES[result_index]} ({result_index})"
                confidence = confidences[result_index] * 100

                # Generate confidence plot
                plot_path = os.path.join(app.config["PLOT_FOLDER"], f"confidence_{file.filename}.png")
                create_confidence_plot([c * 100 for c in confidences], CLASS_NAMES, plot_path)

            return render_template(
                "result.html",
                result=result_text,
                confidence=f"{confidence:.2f}%",
                original_image_path=url_for("static", filename=f"uploads/{file.filename}"),
                predicted_image_path=url_for("static", filename=f"predictions/{os.path.basename(predicted_image_path)}") if predicted_image_path else None,
                plot_path=url_for("static", filename=f"plots/{os.path.basename(plot_path)}") if plot_path else None,
                model_name=model_name,
            )
        except Exception as e:
            # Handle unexpected errors
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
