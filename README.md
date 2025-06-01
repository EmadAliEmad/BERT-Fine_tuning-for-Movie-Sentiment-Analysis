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
