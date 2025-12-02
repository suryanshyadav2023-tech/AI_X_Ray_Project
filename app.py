from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='templates')

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_thickening", "Pneumonia", "Pneumothorax"
]

IMG_SIZE = (224, 224)

def build_model(num_classes):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = build_model(len(CLASS_NAMES))
model.load_weights("model.weights.h5")
print("✅ DenseNet121 model loaded successfully with 14 output classes!")

def prepare_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read()))
        x = prepare_image(img)
        preds = model.predict(x)
        preds = preds.flatten().tolist()

        if len(preds) != len(CLASS_NAMES):
            return jsonify({
                "error": "Model output size mismatch",
                "received": len(preds),
                "expected": len(CLASS_NAMES)
            })

        results = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
