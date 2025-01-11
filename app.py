from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, static_folder="Static", template_folder="Templet")

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "house_price_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
    exit(1)

# Route for the home page
@app.route('/')
def home():
    return render_template("index.html")

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract features
        features = data.get("features")
        if not features:
            return jsonify({"error": "Missing 'features' key in input data"}), 400

        # Predict using the model
        prediction = model.predict([features])
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
