%%writefile app.py
import uvicorn # ASGI server, used to run FastAPI applications.
import tensorflow as tf # Core deep learning library, used for loading and running the model.
from fastapi import FastAPI, Request # FastAPI framework for building web APIs. `Request` is for accessing request details.
from pydantic import BaseModel # Used by FastAPI for data validation and defining API request/response schemas.
from typing import List, Dict, Union # For type hints, ensuring data integrity for lists, dictionaries, and flexible types.
from transformers import AutoTokenizer, TFBertModel # Hugging Face components for tokenizer and BERT model in TensorFlow.
from pathlib import Path # For object-oriented filesystem paths, used for model file paths.
from loguru import logger # Professional logging library for structured output.
from rich.console import Console # From Rich library, for visually appealing console output.
from rich.panel import Panel # From Rich, for creating distinct, framed panels in console output.
import json # For handling JSON data, used for serialization/deserialization.
import re # Regular expression module for text pattern matching (used in cleaning).
import html # Module for decoding HTML entities in text (used in cleaning).

# Import project-specific configurations and the logging setup function.
from config import project_config, model_config # Access project directories and model hyperparameters.
from logger import setup_logging # Function to initialize our logging system.

setup_logging() # Initialize the logging system early, ensuring all subsequent messages are captured.
console = Console() # Create a Rich Console instance for custom formatted prints.

class PredictionRequest(BaseModel):
    """
    Pydantic model defining the expected structure of the request body for the /predict endpoint.
    This ensures that incoming data adheres to a specified format.
    """
    texts: List[str] # A list of strings, where each string is a text for sentiment prediction.

class PredictionResponse(BaseModel):
    """
    Pydantic model defining the expected structure of the response body from the /predict endpoint.
    This provides clear documentation and validation for the API's output.
    """
    predictions: List[Dict[str, Union[str, float]]] # A list of dictionaries, each containing prediction details.

# Initialize the FastAPI application instance.
# Provides metadata like title, description, and version for the API documentation (Swagger UI).
app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="API for classifying movie review sentiment using a fine-tuned BERT model.",
    version="1.0.0",
)

model = None # Global variable to hold the loaded TensorFlow model. Initialized to None.
tokenizer = None # Global variable to hold the loaded BERT tokenizer. Initialized to None.

def clean_text(text: str) -> str:
    """
    Cleans a single input text string by applying a series of preprocessing steps.
    This function's implementation must be IDENTICAL to the `clean_text` function in `data_loader.py`
    to ensure that the text processed during inference matches the format of text during training.
    
    Args:
        text (str): The raw input text string to be cleaned.
        
    Returns:
        str: The cleaned text string.
    """
    if not isinstance(text, str): # Check if the input is actually a string.
        return "" # Return an empty string if the input is not a string, to prevent errors.
    
    text = html.unescape(text) # Decode HTML entities (e.g., & -> &), essential for web-scraped text.
    text = re.sub(r'<[^>]+>', '', text) # Remove any HTML tags found in the text.
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace by replacing multiple spaces/newlines with a single space.
    text = re.sub(r'[^\w\s.,!?;:-]', '', text) # Remove special characters, keeping alphanumeric, whitespace, and basic punctuation.
    text = text.lower().strip() # Convert text to lowercase and remove leading/trailing whitespace.
    return text

@app.on_event("startup")
async def load_model():
    """
    Asynchronous function that runs ONCE when the FastAPI application starts up.
    Its purpose is to load the pre-trained BERT model and its tokenizer into memory.
    This prevents the model from being reloaded for every incoming prediction request,
    which significantly improves API performance.
    """
    global model, tokenizer # Declare these variables as global so they can be accessed and modified outside this function.
    try:
        model_path = project_config.models_dir / "best_model.h5" # Construct the full path to the saved model using ProjectConfig.
        tokenizer_name = model_config.model_name # Get the pre-trained model name for tokenizer initialization from ModelConfig.

        if not model_path.exists(): # Check if the model file actually exists on disk.
            logger.error(f"Model file not found at {model_path}. Please train the model first.")
            console.print(Panel(f"[bold red]❌ Error: Model file not found at {model_path}. Please train the model first.[/bold red]", style="red"))
            return # Exit the function if the model file is missing, preventing further errors.

        console.print(f"[bold blue]Loading model from: {model_path}[/bold blue]")
        # Load the Keras model from the H5 file.
        # `custom_objects={'TFBertModel': TFBertModel}` is CRUCIAL here. It explicitly tells Keras how to
        # reconstruct the `TFBertModel` layer, which is a custom layer from Hugging Face Transformers.
        # This resolves the `ValueError: Unknown layer: 'TFBertModel'` error during model loading.
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'TFBertModel': TFBertModel}, 
            compile=False # Do not compile the model during loading; it will be compiled manually below.
        ) 
        # Manually compile the loaded model. This is necessary to correctly set up the optimizer
        # and metrics, even if we are only using the model for prediction (as model.predict() benefits).
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate), 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                      metrics=['accuracy'])
        console.print("[green]✅ Model loaded successfully![/green]")

        console.print(f"[bold blue]Loading tokenizer: {tokenizer_name}[/bold blue]")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) # Load the tokenizer based on the model name.
        console.print("[green]✅ Tokenizer loaded successfully![/green]")

        logger.info("Model and tokenizer loaded successfully.")

    except Exception as e:
        # Catch and log any general exceptions that occur during model or tokenizer loading.
        logger.error(f"Failed to load model or tokenizer: {e}")
        console.print(Panel(f"[bold red]❌ Failed to load model or tokenizer: {e}[/bold red]", style="red"))
        model = None # Set model and tokenizer to None to indicate they are not ready.
        tokenizer = None

@app.get("/health")
async def health_check():
    """
    Defines a simple GET endpoint at /health.
    This endpoint serves as a health check to verify if the API is running and if
    the model and tokenizer have been successfully loaded into memory.
    
    Returns:
        dict: A dictionary indicating the API status and model/tokenizer loading status.
    """
    return {"status": "ok", "model_loaded": model is not None, "tokenizer_loaded": tokenizer is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Defines a POST endpoint at /predict to receive text inputs and return sentiment predictions.
    It takes a JSON request body containing a list of texts and returns a structured JSON response.
    
    Args:
        request (PredictionRequest): A Pydantic model instance containing the list of texts to predict.
        
    Returns:
        PredictionResponse: A Pydantic model instance containing a list of dictionaries,
                            each with the original text, predicted sentiment label, and confidence score.
    """
    # Check if the model and tokenizer are loaded. If not, return an error response.
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not loaded. Cannot process prediction.")
        return PredictionResponse(predictions=[{"text": t, "sentiment": "Error: Model not loaded"} for t in request.texts])

    try:
        console.print(f"[bold blue]Received {len(request.texts)} texts for prediction.[/bold blue]")
        # Clean the input texts using the same `clean_text` function as during training.
        cleaned_texts = [clean_text(text) for text in request.texts] 
        
        # Tokenize the cleaned input texts.
        # `padding='max_length'` is CRUCIAL here. It ensures that all input sequences are padded
        # to the exact `max_length` (128) defined in the model_config, resolving the
        # `ValueError: Input 0 ... incompatible with the layer: expected shape=(None, 128), found shape=(None, X)` error.
        inputs = tokenizer(
            cleaned_texts,
            padding='max_length',  # Explicitly pads all sequences to `model_config.max_length`.
            truncation=True,       # Truncate sequences longer than `max_length`.
            max_length=model_config.max_length, # Uses `max_length` from config for consistency.
            return_tensors='tf'    # Returns TensorFlow tensors for model input.
        )
        
        # Make predictions using the loaded model.
        predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
        predicted_classes = tf.argmax(predictions, axis=1).numpy() # Get the index of the highest probability (0 or 1).
        confidence_scores = tf.reduce_max(predictions, axis=1).numpy() # Get the probability of the predicted class.

        sentiment_map = {0: "Negative", 1: "Positive"} # Map numerical predictions to human-readable labels.
        results = []
        # Compile results for each input text.
        for i, text in enumerate(request.texts):
            results.append({
                "text": text,
                "sentiment": sentiment_map[predicted_classes[i]],
                "confidence": float(confidence_scores[i]) # Convert NumPy float to standard Python float for JSON serialization.
            })
        logger.info(f"Successfully predicted sentiment for {len(request.texts)} texts.")
        return PredictionResponse(predictions=results) # Return the structured prediction response.

    except Exception as e:
        # Catch and log any errors that occur during the prediction process.
        logger.error(f"Error during prediction: {e}", exc_info=True) # `exc_info=True` logs the full traceback.
        # Return an error message in the response to the client.
        return PredictionResponse(predictions=[{"text": t, "sentiment": f"Error: {e}"} for t in request.texts])