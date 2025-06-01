import requests # Library for making HTTP requests (e.g., GET, POST) to web services/APIs.
import json # For working with JSON data (serializing Python objects to JSON, parsing JSON responses).
import time # For pausing execution (e.g., to wait for the API to start up).
from rich.console import Console # From Rich library, for beautiful and structured console output.
from rich.panel import Panel # From Rich, for displaying visually distinct panels in console output.

console = Console() # Initialize a Rich Console instance for formatted printing to the notebook output.

# ====== IMPORTANT: Update this URL with the actual Ngrok Tunnel URL displayed by the previous cell. ======
# The ngrok URL is temporary and changes each time the tunnel is established (i.e., every time Cell 14 is run).
# You MUST copy the NEW URL from the output of Cell 14 and paste it here before executing this cell.
ngrok_url = "https://f085-146-148-39-26.ngrok-free.app" # Placeholder: REPLACE THIS LINE with the NEW Ngrok URL!
# Example of finding the URL: Look for "Ngrok Tunnel URL:" in the output of the previous cell (Cell 14).
# =========================================================================================================

console.print(f"[bold blue]Testing API at: {ngrok_url}[/bold blue]") # Inform the user about the API URL being tested.

# Pause execution to give the FastAPI application (running via Uvicorn and Ngrok)
# sufficient time to fully start up and load the BERT model.
# This duration might need adjustment based on model size, internet speed, and Kaggle's GPU/CPU speed.
time.sleep(20) 

try:
    # Send a GET request to the /health endpoint of the API.
    # This checks if the API is running and if the model/tokenizer are loaded.
    health_response = requests.get(f"{ngrok_url}/health")
    console.print(f"Health Check Status: {health_response.status_code}") # Print the HTTP status code (e.g., 200 OK).
    console.print(f"Health Check Response: {health_response.json()}") # Print the JSON response from the health check.
    
    # Extract the 'model_loaded' status from the health check response.
    model_loaded = health_response.json().get("model_loaded", False)
    if not model_loaded:
        # If the model is not reported as loaded by the health check, display a warning.
        console.print(Panel("[bold yellow]Warning: Model not yet loaded according to health check. Prediction might fail.[/bold yellow]", style="yellow"))
except requests.exceptions.ConnectionError as e:
    # Catch a ConnectionError, which means the API could not be reached.
    # This often indicates that ngrok or Uvicorn is not running, or the URL is incorrect.
    console.print(Panel(f"[bold red]Error: Could not connect to the API. Make sure ngrok and uvicorn are running. Error: {e}[/bold red]", style="red"))
    health_response = None # Set response to None to prevent further processing.
    model_loaded = False # Indicate that the model is not loaded due to connection failure.
except json.JSONDecodeError as e: 
    # Catch a JSONDecodeError. This means the API responded, but its content was not valid JSON.
    # This can happen if the API returns an HTML error page (e.g., 404 Not Found) instead of JSON.
    console.print(Panel(f"[bold red]Error: Could not decode JSON response from health check. Error: {e}[/bold red]", style="red"))
    health_response = None
    model_loaded = False


# Proceed to send prediction requests only if the health check passed (status 200)
# and the model was confirmed as loaded.
if health_response and health_response.status_code == 200 and model_loaded:
    # Define a list of sample texts to send to the API for sentiment prediction.
    test_texts = [
        "This movie was absolutely fantastic! I loved every moment of it.",
        "The plot was confusing and the acting was terrible. A complete waste of time.",
        "It was an okay movie, nothing special, but not bad either.",
        "What a masterpiece! Highly recommend."
    ]

    headers = {"Content-Type": "application/json"} # Set HTTP header to indicate the request body is JSON.
    data = {"texts": test_texts} # Prepare the request body as a Python dictionary.

    console.print("\n[bold blue]Sending prediction request...[/bold blue]") # Inform the user about sending the prediction request.
    try:
        # Send a POST request to the /predict endpoint.
        # `data=json.dumps(data)` converts the Python dictionary to a JSON string for the request body.
        predict_response = requests.post(f"{ngrok_url}/predict", headers=headers, data=json.dumps(data))
        console.print(f"Prediction Status: {predict_response.status_code}") # Print the HTTP status code for prediction.
        console.print("Prediction Response:") # Label for the prediction response.
        console.print(json.dumps(predict_response.json(), indent=2)) # Parse and print the JSON prediction response, pretty-printed.
        console.print(Panel("[bold green]✅ Prediction successful![/bold green]")) # Confirm successful prediction.
    except requests.exceptions.ConnectionError as e:
        # Handle connection errors during the prediction request.
        console.print(Panel(f"[bold red]Error: Could not connect to the prediction endpoint. Error: {e}[/bold red]", style="red"))
    except Exception as e:
        # Catch any other general errors during the prediction request.
        console.print(Panel(f"[bold red]Error during prediction request: {e}[/bold red]", style="red"))
else:
    # If the API is not ready (e.g., model not loaded, health check failed), inform the user.
    console.print(Panel("[bold red]API is not fully ready for predictions. Check previous output for errors.[/bold red]", style="red"))