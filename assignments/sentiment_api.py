
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification using trained NLP model",
    version="1.0.0"
)

# Load model at startup
try:
    model = joblib.load('best_text_pipeline.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Request model
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

# Response model
class PredictionOutput(BaseModel):
    text: str
    prediction: int
    sentiment: str
    confidence: float = None

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Single prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(input_data: TextInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Make prediction
        prediction = model.predict([input_data.text])[0]

        # Get probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([input_data.text])[0]
            confidence = float(max(proba))

        sentiment = "positive" if prediction == 1 else "negative"

        return PredictionOutput(
            text=input_data.text,
            prediction=int(prediction),
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(input_data: BatchTextInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        predictions = model.predict(input_data.texts)

        # Get probabilities if available
        confidences = None
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(input_data.texts)
            confidences = [float(max(p)) for p in probas]

        results = []
        for i, (text, pred) in enumerate(zip(input_data.texts, predictions)):
            sentiment = "positive" if pred == 1 else "negative"
            confidence = confidences[i] if confidences else None
            results.append(PredictionOutput(
                text=text,
                prediction=int(pred),
                sentiment=sentiment,
                confidence=confidence
            ))

        return BatchPredictionOutput(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
