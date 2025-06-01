# 🚀 Production-Ready BERT Sentiment Analysis API

An end-to-end sentiment analysis solution built with a fine-tuned BERT model, showcasing a robust training pipeline and a high-performance FastAPI for real-time inference. This project demonstrates practical application of Natural Language Processing (NLP), Deep Learning, and MLOps principles.

## ✨ Key Features:

*   **End-to-End Pipeline:** Covers data loading, cleaning, preprocessing, model training, evaluation, and API deployment.
*   **BERT-based Model:** Utilizes `bert-base-uncased` from Hugging Face Transformers for state-of-the-art sentiment classification.
*   **Modular Design:** Organized into separate Python modules (`config.py`, `data_loader.py`, `model.py`, `trainer.py`, `evaluator.py`, `app.py`) for clarity and maintainability.
*   **Professional Training:** Implements advanced Keras callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard) for efficient and stable model training.
*   **Comprehensive Evaluation:** Provides detailed performance metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and generates interactive visualizations (Training History, Confusion Matrix, ROC Curve).
*   **Real-time Inference API:** Deployed using FastAPI and Uvicorn, enabling sentiment predictions via HTTP requests.
*   **Robust Error Handling:** Demonstrates strong problem-solving capabilities by addressing common environmental and library compatibility challenges.

## 📚 Technologies Used:

*   **Deep Learning Frameworks:** `TensorFlow`, `Keras`
*   **NLP:** `Hugging Face Transformers` (`TFBertModel`, `AutoTokenizer`)
*   **Data Manipulation:** `Pandas`, `NumPy`
*   **Machine Learning Utilities:** `Scikit-learn`
*   **API Development:** `FastAPI`, `Uvicorn`, `Pydantic`
*   **Logging & Visualization:** `Loguru`, `Rich`, `Plotly`, `Matplotlib`, `Seaborn`
*   **Environment Management:** `Python venv`
*   **Version Control:** `Git`, `GitHub`

## 📂 Project Structure:

```bash
.
├── app.py                       # FastAPI application for inference API.
├── config.py                    # Centralized configuration for model, training, and project paths.
├── data_loader.py               # Handles IMDB dataset loading, text cleaning, and data splitting.
├── evaluator.py                 # Model evaluation suite and visualization generation.
├── logger.py                    # Professional logging setup with Loguru and Rich.
├── main.py                      # Main script for model training and evaluation pipeline.
├── model.py                     # Defines BERT model architecture.
├── trainer.py                   # Encapsulates the model training pipeline.
├── requirements.txt             # List of all Python dependencies for the project.
├── API Images Output Test/      # Directory to store screenshots for README.
└── .gitignore                   # Specifies files/directories to be ignored by Git (e.g., large models, virtual environments).

## ⚙️ How to Run Locally:

This project's model (`best_model.h5`) was trained on Kaggle's GPU-enabled notebooks. The trained model file is **not directly included** in this GitHub repository due to its large size (approx. 440MB) and GitHub's file size limits. However, it can be downloaded directly from the Kaggle Notebook's output.



**Setup Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EmadAliEmad//BERT-Fine_tuning-for-Movie-Sentiment-Analysis.git
    cd BERT-Fine_tuning-for-Movie-Sentiment-Analysis
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # Using Python 3.11 directly (if installed as 'python3.11' or 'py -3.11')
    py -3.11 -m venv venv_api 
    # Or simply 'python -m venv venv_api' if 3.11 is your default Python.
    
    # Activate the virtual environment:
    .\venv_api\Scripts\activate   # On Windows (Command Prompt or PowerShell)
    source venv_api/bin/activate  # On macOS/Linux/Git Bash
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   *(Ignore any `WARNING` messages about dependency conflicts unless they are `ERROR` preventing installation.)*

4.  **Run the FastAPI application locally:**
    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will now be running on `http://127.0.0.1:8000`. Keep this terminal open.

## 🧪 Testing the API:

Once the API is running, open your web browser and navigate to: `http://127.0.0.1:8000/docs`

You will see the interactive Swagger UI where you can test the endpoints.

**1. API Interface (Swagger UI):**
![API Interface](API%20Images%20Output%20Test/API%20Interface.png)
*A view of the interactive Swagger UI for the FastAPI endpoints.*

**2. Health Check Endpoint Output:**
![API Init](API%20Images%20Output%20Test/API%20init.png)
*Successful health check indicating the model and tokenizer are loaded upon API startup.*

**3. Prediction Endpoint Output:**
![API Result](API%20Images%20Output%20Test/API%20result.png)
*Screenshot showing successful sentiment predictions for sample texts from the /predict endpoint.*



## 📊 Results & Performance:

The model achieved strong performance on the IMDB test set (using 10,000 samples).
*   **Accuracy:** ~0.87
*   **ROC AUC:** ~0.94


## 🚀 Future Enhancements:

*   Integrate a user-friendly web interface (e.g., with Streamlit or Flask).
*   Explore advanced fine-tuning techniques or larger BERT variants (`bert-large-uncased`).
*   Implement more sophisticated text preprocessing or post-processing (e.g., emotion detection, aspect-based sentiment).
*   Containerize the application using Docker for easier deployment.
*   Deploy the API to a cloud platform (e.g., Hugging Face Spaces, AWS Lambda, Google Cloud Run).


