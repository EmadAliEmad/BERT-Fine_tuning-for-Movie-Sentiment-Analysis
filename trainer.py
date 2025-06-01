%%writefile trainer.py
import tensorflow as tf # Core deep learning library for model training.
from tensorflow.keras.optimizers import Adam # Optimizer for updating model weights.
from tensorflow.keras.losses import SparseCategoricalCrossentropy # Loss function for integer labels.
from tensorflow.keras.metrics import SparseCategoricalAccuracy # Metric to track accuracy for integer labels.
from tensorflow.keras.callbacks import (
    ModelCheckpoint,       # Callback to save the best model during training.
    EarlyStopping,         # Callback to stop training early if validation metric plateaus.
    ReduceLROnPlateau,     # Callback to reduce learning rate when a metric stops improving.
    TensorBoard,           # Callback for visualizing training progress.
    CSVLogger              # Callback to save training history to a CSV file.
)
from sklearn.metrics import classification_report, confusion_matrix # For detailed evaluation metrics.
import numpy as np # For numerical operations, especially converting labels to NumPy arrays.
import matplotlib.pyplot as plt # For plotting (though mostly handled by Plotly for interactive plots).
import seaborn as sns # For enhanced statistical data visualization (often used with Matplotlib).
from typing import Dict, List, Tuple, Optional # For type hinting.
from loguru import logger # For structured logging.
import json # For saving training history and configurations to JSON files.
import time # For measuring training time.

# Import configuration objects from config.py.
from config import ModelConfig, TrainingConfig, project_config

class BERTTrainer:
    """
    Professional BERT training pipeline.
    Manages model compilation, callbacks setup, and the training loop.
    """
    
    def __init__(self, 
                 model: tf.keras.Model,          # The Keras model to be trained.
                 model_config: ModelConfig,       # Model configuration object.
                 training_config: TrainingConfig): # Training configuration object.
        # Call the constructor of the parent class (in this case, 'object' implicitly, or another base if present).
        # This resolves a TypeError if `super().__init__(**kwargs)` was used without `kwargs` being passed.
        super().__init__() 
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.history = None # To store training history (loss, accuracy per epoch).
        self.tokenizer = None # To store the BERT tokenizer.

        # Initialize tokenizer from Hugging Face Transformers.
        # Imported here (inside __init__) to avoid potential circular import issues,
        # ensuring the tokenizer is available when the trainer is instantiated.
        from transformers import BertTokenizer 
        self.tokenizer = BertTokenizer.from_pretrained(model_config.model_name)

        logger.info("BERT Trainer initialized")
    
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> tf.data.Dataset:
        """
        Prepares a TensorFlow dataset from texts and labels, including tokenization.
        Note: This method is currently NOT directly used in the `train` method's workflow
        (tokenization is handled directly within `train`). It might be a remnant
        from an alternative data pipeline design or for future flexibility.
        
        Args:
            texts (List[str]): List of raw text samples.
            labels (List[int]): List of corresponding numerical labels.

        Returns:
            tf.data.Dataset: A TensorFlow dataset, processed and ready for model input.
        """
        def tokenize_function(texts, labels):
            # Tokenize texts, converting them to numerical IDs and attention masks.
            tokenized = self.tokenizer(
                texts.numpy().tolist(), # Convert TensorFlow tensors back to Python list for tokenizer processing.
                padding=True,           # Pad sequences to a uniform length.
                truncation=True,        # Truncate sequences that are too long.
                max_length=self.model_config.max_length, # Use max_length from model config.
                return_tensors='tf'     # Ensure output is in TensorFlow tensor format.
            )
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }, labels
        
        # Create a TensorFlow dataset from input texts and labels.
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        
        # The following lines (`if shuffle:` and `dataset = dataset.batch(...)`)
        # contain variables (`shuffle`, `batch_size`) that are not defined in this method's scope.
        # If this method were to be fully utilized, these would need to be passed as arguments or accessed from config.
        # These lines are part of the original code, but their functionality relies on external scope.
        if shuffle: # This `shuffle` variable is not defined in this method's scope.
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        dataset = dataset.batch(batch_size) # This `batch_size` variable is not defined in this method's scope.
        dataset = dataset.prefetch(tf.data.AUTOTUNE) # Optimize data loading by prefetching batches.
        return dataset

    def compile_model(self):
        """
        Compiles the Keras model with a specified optimizer, loss function, and metrics.
        This prepares the model for the training process before fitting the data.
        """
        optimizer = Adam(learning_rate=self.model_config.learning_rate) # Use the Adam optimizer with the configured learning rate.
        # Define the loss function for multi-class classification with integer labels.
        # `from_logits=False` because our model's final layer uses `softmax` activation (outputs probabilities).
        loss = SparseCategoricalCrossentropy(from_logits=False) 
        metrics = [SparseCategoricalAccuracy(name='accuracy')] # Track sparse categorical accuracy during training.
        
        # Compile the model with the defined optimizer, loss, and metrics.
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        logger.info("Model compiled successfully")
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]: # Added `self` as the first argument.
        """
        Sets up a list of Keras Callbacks to enhance, monitor, and control the training process.
        This includes saving the best model, early stopping, learning rate reduction, and logging.
        
        Returns:
            List[tf.keras.callbacks.Callback]: A list of configured Keras callback instances.
        """
        callbacks = []
        
        # ModelCheckpoint: Saves the model's weights or entire model at specific points during training.
        checkpoint_path = project_config.models_dir / "best_model.h5" # Define the path where the best model will be saved.
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path), # File path for saving the model.
            monitor='val_accuracy',        # Metric to monitor for improvement (validation accuracy).
            save_best_only=True,           # Only save the model if the monitored metric improves.
            save_weights_only=False,       # Save the entire model (architecture + weights), not just weights.
            mode='max',                    # 'max' indicates that a higher 'val_accuracy' is considered better.
            verbose=1                      # Display messages when saving.
        )
        callbacks.append(checkpoint)
        
        # EarlyStopping: Stops training automatically if the monitored metric stops improving for a specified number of epochs.
        early_stopping = EarlyStopping(
            monitor='val_accuracy',        # Monitor validation accuracy.
            patience=2,                    # Number of epochs to wait for improvement before stopping.
            restore_best_weights=True,     # Reverts model weights to the best performing epoch.
            mode='max',                    # 'max' indicates that a higher 'val_accuracy' is better.
            verbose=1                      # Display messages when early stopping is triggered.
        )
        callbacks.append(early_stopping)
        
        # ReduceLROnPlateau: Reduces the learning rate when the monitored metric plateaus (stops improving).
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', # Monitor validation loss.
            factor=0.5,         # Factor by which the learning rate will be reduced (new_lr = old_lr * factor).
            patience=1,         # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=1e-7,        # Lower bound on the learning rate.
            mode='min',         # 'min' indicates that a lower 'val_loss' is better.
            verbose=1           # Display messages when learning rate is reduced.
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard: A visualization toolkit for TensorFlow to inspect training runs, model graphs, and histograms.
        tensorboard = TensorBoard(
            log_dir=str(project_config.logs_dir / "tensorboard"), # Directory for TensorBoard logs.
            histogram_freq=1, # Compute histograms for layer activations and weights every epoch.
            write_graph=True, # Visualize the model's computation graph.
            update_freq='epoch' # How often TensorBoard logs are updated (e.g., after each epoch).
        )
        callbacks.append(tensorboard)
        
        # CSVLogger: Streams epoch results (loss, metrics) to a CSV file for easy analysis.
        csv_logger = CSVLogger(
            str(project_config.logs_dir / "training_log.csv"), # Path to the CSV log file.
            append=True # If True, new training runs' results are appended to the file if it already exists.
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(self, 
              train_data: Dict,     # Dictionary containing training texts and their corresponding labels.
              val_data: Dict) -> tf.keras.callbacks.History: # Dictionary containing validation texts and their labels.
        """
        Executes the training loop for the BERT model using the prepared data.
        The model is fitted on the training data and validated on the validation data.
        
        Args:
            train_data (Dict): A dictionary with 'texts' (list of strings) and 'labels' (list of integers)
                                for the training set.
            val_data (Dict): A dictionary with 'texts' (list of strings) and 'labels' (list of integers)
                             for the validation set.
            
        Returns:
            tf.keras.callbacks.History: A Keras History object, containing records of training loss values
                                        and metrics values at successive epochs.
        """
        logger.info("Starting model training...")
        start_time = time.time() # Record the start time to calculate total training duration.
        
        # Extract text lists and label lists from the input data dictionaries.
        train_texts = train_data['texts']
        train_labels = train_data['labels']
        val_texts = val_data['texts']
        val_labels = val_data['labels']
        
        # Tokenize training data using the BERT tokenizer.
        logger.info("Tokenizing training data...")
        train_encodings = self.tokenizer(
            train_texts,
            padding=True,                  # Pads sequences to `max_length` or the longest in the batch.
            truncation=True,               # Truncate sequences longer than `max_length`.
            max_length=self.model_config.max_length, # Uses the `max_length` specified in ModelConfig.
            return_tensors='tf'            # Returns TensorFlow tensors for input_ids and attention_mask.
        )
        
        # Tokenize validation data using the BERT tokenizer.
        logger.info("Tokenizing validation data...")
        val_encodings = self.tokenizer(
            val_texts,
            padding=True,
            truncation=True,
            max_length=self.model_config.max_length,
            return_tensors='tf'
        )
        
        # Compile the model (if it hasn't been compiled yet) using defined optimizer, loss, and metrics.
        self.compile_model()
        
        # Setup and retrieve the list of Keras callbacks for this training run.
        callbacks = self.setup_callbacks()
        
        # Start the actual model training process.
        self.history = self.model.fit(
            x=[train_encodings['input_ids'], train_encodings['attention_mask']], # BERT requires input_ids and attention_mask.
            y=np.array(train_labels), # Convert labels to a NumPy array.
            validation_data=( # Provide validation data for monitoring training progress.
                [val_encodings['input_ids'], val_encodings['attention_mask']], 
                np.array(val_labels)
            ), 
            epochs=self.model_config.epochs, # Number of epochs from ModelConfig.
            batch_size=self.model_config.batch_size, # Batch size from ModelConfig.
            callbacks=callbacks, # Apply the configured callbacks during training.
            verbose=1 # Display training progress in detail for each epoch.
        )
        
        training_time = time.time() - start_time # Calculate the total time taken for training.
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.history # Return the training history object.
    
    def save_training_artifacts(self):
        """
        Saves important artifacts from the training process:
        - The training history (loss and metrics per epoch) as a JSON file.
        - The model configuration (ModelConfig and TrainingConfig) as a JSON file.
        This is crucial for reproducibility and analysis of past training runs.
        """
        
        # Save training history if the `history` object is available (meaning training occurred).
        if self.history:
            history_path = project_config.output_dir / "training_history.json"
            
            # Prepare the training history for JSON serialization.
            # This step converts NumPy/TensorFlow float32 values (which are not directly JSON serializable)
            # into standard Python floats. This resolved the `TypeError: Object of type float32 is not JSON serializable`
            # error that occurred when attempting to save the history dictionary.
            serializable_history = {}
            for key, value_list in self.history.history.items():
                serializable_history[key] = [float(val) for val in value_list] # Convert each float32 value in lists to Python float.
            
            # Write the serializable history to a JSON file with pretty printing (indent=2).
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2) 
            logger.info(f"Training history saved to {history_path}")
        
        # Save the combined model and training configurations.
        config_path = project_config.output_dir / "model_config.json"
        config_dict = {
            'model_config': self.model_config.__dict__, # Convert ModelConfig dataclass instance to a dictionary.
            'training_config': self.training_config.__dict__ # Convert TrainingConfig dataclass instance to a dictionary.
        }
        # Write the configuration dictionary to a JSON file.
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Model configuration saved to {config_path}")