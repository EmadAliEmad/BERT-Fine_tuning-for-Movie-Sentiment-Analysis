%%writefile evaluator.py
import numpy as np # For numerical operations and array manipulation.
import matplotlib.pyplot as plt # Basic plotting functions (used by seaborn).
import seaborn as sns # High-level statistical plotting library.
from sklearn.metrics import ( # Various metrics for model evaluation.
    accuracy_score,           # Overall classification accuracy.
    precision_score,          # Precision for positive and negative classes.
    recall_score,             # Recall for positive and negative classes.
    f1_score,                 # F1-score (harmonic mean of precision and recall).
    classification_report,    # Detailed report of precision, recall, f1-score for each class.
    confusion_matrix,         # Matrix showing correct and incorrect predictions.
    roc_auc_score,            # Area Under the Receiver Operating Characteristic (ROC) Curve.
    roc_curve                 # Data points for plotting the ROC curve.
)
import plotly.graph_objects as go # For creating interactive graph objects (e.g., scatter plots).
import plotly.express as px # Simplified interface for Plotly for quick plots.
from plotly.subplots import make_subplots # For creating subplots in Plotly.
from typing import Dict, List, Tuple, Optional # For type hints, improving code readability.
from loguru import logger # For structured and informative logging.

# Crucial import: TensorFlow is needed for Keras model type hints and model operations.
import tensorflow as tf 
# Import project and model configurations from config.py.
from config import project_config, model_config 

class ModelEvaluator:
    """
    Comprehensive model evaluation suite.
    This class is responsible for making predictions, calculating various performance metrics,
    and generating interactive visualizations of the model's performance.
    """
    
    # The __init__ method is designed to directly receive the already initialized model and tokenizer.
    # This prevents redundant loading/initialization of the tokenizer, which resolves issues like
    # `NameError: name 'tf' is not defined` and `HFValidationError` seen in earlier iterations.
    def __init__(self, model: tf.keras.Model, tokenizer): # The trained Keras model, and the pre-trained tokenizer.
        self.model = model
        self.tokenizer = tokenizer # Stores the tokenizer provided (e.g., from BERTTrainer).
        self.class_names = ['Negative', 'Positive'] # Defines human-readable class names for reports and plots.
        logger.info("Model Evaluator initialized with provided tokenizer") # Log confirmation of initialization.
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes sentiment predictions on a list of raw text strings using the loaded BERT model.
        The texts are tokenized, then passed through the model.
        
        Args:
            texts (List[str]): A list of text strings for which sentiment predictions are required.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - predictions (np.ndarray): The raw probability scores output by the model for each class.
                - predicted_classes (np.ndarray): The numerical class label (0 or 1) with the highest probability.
        """
        # Tokenize the input texts using the tokenizer.
        # This converts text into numerical input IDs and attention masks expected by BERT.
        encodings = self.tokenizer(
            texts,
            padding=True,              # Pads sequences to `max_length` or the longest in the batch.
            truncation=True,           # Truncates sequences longer than `max_length`.
            max_length=model_config.max_length, # Uses the `max_length` specified in ModelConfig for consistent input shape.
            return_tensors='tf'        # Ensures the output is in TensorFlow tensor format.
        )
        
        # Get predictions from the model by passing the tokenized inputs.
        # The model expects separate tensors for input_ids and attention_mask.
        predictions = self.model.predict([
            encodings['input_ids'],
            encodings['attention_mask']
        ])
        
        # Determine the predicted class by finding the index of the highest probability.
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predictions, predicted_classes # Return raw probabilities and the predicted class labels.
    
    def evaluate_model(self, test_data: Dict) -> Dict:
        """
        Performs a comprehensive evaluation of the model's performance on the test data.
        It calculates various standard metrics and generates a detailed classification report
        and confusion matrix.
        
        Args:
            test_data (Dict): A dictionary containing 'texts' (list of strings) and 'labels' (list of integers)
                              for the test dataset.
            
        Returns:
            Dict: A dictionary containing:
                - 'metrics': A dictionary of scalar performance metrics (accuracy, precision, recall, f1-score, ROC AUC).
                - 'classification_report': A detailed report per class (precision, recall, f1-score, support).
                - 'confusion_matrix': A NumPy array representing the confusion matrix.
                - 'predictions': Raw probability predictions from the model.
                - 'predicted_classes': The numerical class labels predicted by the model.
                - 'true_labels': The actual numerical class labels from the test data.
        """
        logger.info("Starting model evaluation...")
        
        test_texts = test_data['texts'] # Extract test texts.
        test_labels = np.array(test_data['labels']) # Extract true labels and convert to NumPy array.
        
        # Get predictions (probabilities and predicted class labels) for the test texts.
        probabilities, predicted_classes = self.predict(test_texts)
        
        # Calculate key performance metrics using scikit-learn.
        metrics = {
            'accuracy': accuracy_score(test_labels, predicted_classes), # Overall accuracy.
            'precision': precision_score(test_labels, predicted_classes, average='weighted'), # Weighted average precision.
            'recall': recall_score(test_labels, predicted_classes, average='weighted'),     # Weighted average recall.
            'f1_score': f1_score(test_labels, predicted_classes, average='weighted'),       # Weighted average F1-score.
            'roc_auc': roc_auc_score(test_labels, probabilities[:, 1]) # ROC AUC score for the positive class.
        }
        
        # Generate a detailed classification report, using human-readable class names.
        class_report = classification_report(
            test_labels, predicted_classes, 
            target_names=self.class_names, # Use 'Negative', 'Positive' for clarity.
            output_dict=True # Return the report as a dictionary.
        )
        
        # Compute the confusion matrix.
        cm = confusion_matrix(test_labels, predicted_classes)
        
        # Compile all evaluation results into a single dictionary.
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': probabilities,
            'predicted_classes': predicted_classes,
            'true_labels': test_labels
        }
        
        logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}") # Log final accuracy.
        return results
    
    def plot_training_history(self, history: tf.keras.callbacks.History):
        """
        Generates and saves an interactive Plotly graph visualizing the model's training history,
        including training and validation accuracy and loss over epochs.
        
        Args:
            history (tf.keras.callbacks.History): The history object returned by `model.fit()`,
                                                  containing epoch-wise training metrics.
        """
        
        # Create a subplot figure with two columns for accuracy and loss.
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Model Accuracy', 'Model Loss'], # Titles for each subplot.
            specs=[[{"secondary_y": False}, {"secondary_y": False}]] # Standard layout.
        )
        
        epochs = range(1, len(history.history['accuracy']) + 1) # Generate x-axis values for epochs.
        
        # Add a trace for Training Accuracy.
        fig.add_trace(
            go.Scatter(
                x=list(epochs), y=history.history['accuracy'], # x-axis: epochs, y-axis: accuracy values.
                mode='lines+markers', name='Training Accuracy', # Display lines and markers.
                line=dict(color='#1f77b4', width=3) # Custom line color and width.
            ),
            row=1, col=1 # Position this trace in the first subplot.
        )
        
        # Add a trace for Validation Accuracy.
        fig.add_trace(
            go.Scatter(
                x=list(epochs), y=history.history['val_accuracy'],
                mode='lines+markers', name='Validation Accuracy',
                line=dict(color='#ff7f0e', width=3)
            ),
            row=1, col=1
        )
        
        # Add a trace for Training Loss.
        fig.add_trace(
            go.Scatter(
                x=list(epochs), y=history.history['loss'],
                mode='lines+markers', name='Training Loss',
                line=dict(color='#1f77b4', width=3)
            ),
            row=1, col=2 # Position this trace in the second subplot.
        )
        
        # Add a trace for Validation Loss.
        fig.add_trace(
            go.Scatter(
                x=list(epochs), y=history.history['val_loss'],
                mode='lines+markers', name='Validation Loss',
                line=dict(color='#ff7f0e', width=3)
            ),
            row=1, col=2
        )
        
        # Update the overall layout of the Plotly figure.
        fig.update_layout(
            title="Training History",
            height=500,
            showlegend=True,
            template="plotly_white" # Use a clean, white-themed template.
        )
        
        # Update axis titles for clarity.
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        # Save the interactive plot as an HTML file in the specified output directory.
        fig.write_html(str(project_config.output_dir / "training_history.html"))
        # fig.show() # This line is commented out as `fig.show()` creates pop-up windows which are generally
                   # not supported or desirable in non-interactive notebook environments like Kaggle.
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """
        Generates and saves an interactive Plotly heatmap visualization of the confusion matrix.
        This helps in understanding the types of errors the model makes (e.g., false positives, false negatives).
        
        Args:
            cm (np.ndarray): The confusion matrix array, typically a 2x2 matrix for binary classification.
        """
        
        # Create a Plotly Express imshow (heatmap) figure.
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"), # Labels for axes and color bar.
            x=self.class_names, # Labels for the predicted classes on the x-axis.
            y=self.class_names, # Labels for the actual classes on the y-axis.
            text_auto=True,     # Automatically display the value of each cell on the heatmap.
            aspect="auto",      # Adjusts the aspect ratio of the heatmap automatically.
            color_continuous_scale="Blues" # Use a blue color scale for the heatmap.
        )
        
        fig.update_layout(
            title="Confusion Matrix", # Set the title of the plot.
            width=500,                # Set the width of the plot.
            height=500                # Set the height of the plot.
        )
        
        # Save the interactive confusion matrix plot as an HTML file.
        fig.write_html(str(project_config.output_dir / "confusion_matrix.html"))
        # fig.show() # Commented out for Notebook compatibility.
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Generates and saves an interactive Plotly plot of the Receiver Operating Characteristic (ROC) curve.
        The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier
        as its discrimination threshold is varied. AUC (Area Under the Curve) provides a single metric
        to summarize the overall performance.
        
        Args:
            y_true (np.ndarray): True binary labels (e.g., 0s and 1s).
            y_proba (np.ndarray): Predicted probabilities of the positive class (e.g., probabilities for class 1).
        """
        
        # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve.
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1]) # Compute ROC for the positive class (column 1 of probabilities).
        # Calculate the Area Under the ROC Curve (AUC score).
        auc_score = roc_auc_score(y_true, y_proba[:, 1])
        
        fig = go.Figure() # Create a new Plotly Figure object.
        
        # Add the main ROC Curve trace to the figure.
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,                                 # FPR on x-axis, TPR on y-axis.
            mode='lines',                                 # Connect points with lines.
            name=f'ROC Curve (AUC = {auc_score:.3f})',    # Label including the calculated AUC score.
            line=dict(color='#1f77b4', width=3)           # Custom line style.
        ))
        
        # Add a diagonal line representing a random classifier (AUC = 0.5).
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],                           # Diagonal line from (0,0) to (1,1).
            mode='lines',
            name='Random Classifier',                     # Label for the random classifier.
            line=dict(color='red', dash='dash')           # Dashed red line.
        ))
        
        # Update the layout of the ROC curve plot.
        fig.update_layout(
            title='ROC Curve',                           # Set the title of the plot.
            xaxis_title='False Positive Rate',           # Label for the x-axis.
            yaxis_title='True Positive Rate',            # Label for the y-axis.
            template="plotly_white"                      # Use a clean, white-themed template.
        )
        
        # Save the interactive ROC curve plot as an HTML file.
        fig.write_html(str(project_config.output_dir / "roc_curve.html"))
        # fig.show() # Commented out for Notebook compatibility.