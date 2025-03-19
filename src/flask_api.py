import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from tensorflow.keras.models import load_model

from process_data import preprocess_text, get_model_embedding, get_sentiment_and_count, get_topics, get_processed_features

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app, version="1.0", title="Car Issue Classification API", description="An API for classifying car issues using machine learning.")

ns = api.namespace("api", description="Endpoints")

# Load the trained model
custom_model = load_model("models/best_model.h5")

# Load available car models
with open("models/params/models_params.json", "r") as file:
    car_models = json.load(file)["models"]

# Define request model for prediction
predict_model = api.model("PredictionRequest", {
    "summary": fields.String(required=True, description="Summary of the car issue"),
    "model": fields.String(required=True, description="Car model name")
})

# Define response model for prediction
predict_response = api.model("PredictionResponse", {
    "summary": fields.String(description="Original summary"),
    "model": fields.String(description="Car model name"),
    "predicted_categories": fields.List(fields.String, description="Predicted issue categories"),
    "probabilities": fields.List(fields.Float, description="Prediction probabilities for each category")
})

# Define request model for sentiment analysis
sentiment_model = api.model("SentimentRequest", {
    "summary": fields.String(required=True, description="Summary of the car issue")
})

# Define response model for sentiment analysis
sentiment_response = api.model("SentimentResponse", {
    "summary": fields.String(description="Original summary"),
    "sentiment": fields.String(description="Sentiment classification"),
    "word_count": fields.Integer(description="Word count in summary")
})

def get_parameters(df):
    """Process text and extract necessary features for model prediction."""
    df["processed_summary"] = df["summary"].apply(preprocess_text)
    df["summary_embedding"] = df["summary"].apply(get_model_embedding)
    df["model_embedding"] = df["Model"].apply(get_model_embedding)

    df = get_sentiment_and_count(df)
    df = get_topics(df)

    embeddings, additional_features = get_processed_features(df)

    return embeddings, additional_features

@ns.route("/health")
class HealthCheck(Resource):
    @api.doc(description="Check if the API is running.")
    def get(self):
        return {"status": "ok"}, 200

@ns.route("/models")
class AvailableModels(Resource):
    @api.doc(description="Get the list of available car models.")
    def get(self):
        return {"available_models": car_models}

@ns.route("/predict")
class Predict(Resource):
    @api.expect(predict_model)
    @api.marshal_with(predict_response)
    @api.doc(description="Predict car issue categories based on summary and model.")
    def post(self):
        input_data = request.json
        text = input_data.get("summary")
        model_name = input_data.get("model")

        if not text or not model_name:
            api.abort(400, "Both 'summary' and 'model' must be provided.")

        df = pd.DataFrame({"summary": [text], "Model": [model_name]})
        embeddings, additional_features = get_parameters(df)

        # Make predictions
        prediction = custom_model.predict({
            "embedding_input": embeddings,
            "additional_input": additional_features
        })

        categories = ["Engine and Performance", "Electrical System", "Other Problems", "Structure and Control", "Safety and Brakes"]
        predicted_categories = [categories[i] for i, prob in enumerate(prediction[0]) if prob > 0.5]

        return {
            "summary": text,
            "model": model_name,
            "predicted_categories": predicted_categories,
            "probabilities": prediction[0].tolist()
        }

@ns.route("/sentiment")
class SentimentAnalysis(Resource):
    @api.expect(sentiment_model)
    @api.marshal_with(sentiment_response)
    @api.doc(description="Analyze the sentiment of a given summary.")
    def post(self):
        input_data = request.json
        text = input_data.get("summary")

        if not text:
            api.abort(400, "The field 'summary' is required.")

        df = pd.DataFrame({"summary": [text]})
        df["processed_summary"] = df["summary"].apply(preprocess_text)
        df = get_sentiment_and_count(df)

        return {
            "summary": text,
            "sentiment": df["sentiment"].iloc[0],
            "word_count": df["word_count"].iloc[0]
        }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
