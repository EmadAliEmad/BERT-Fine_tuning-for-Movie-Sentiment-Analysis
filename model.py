%%writefile model.py
import tensorflow as tf # Core deep learning library for building and training models.
import tensorflow_hub as hub # For reusable machine learning modules.
from transformers import TFBertModel, BertTokenizer # Hugging Face BERT model and tokenizer for TensorFlow.
from typing import Dict, Any, Optional # For type hinting.
from loguru import logger # For logging messages.

# Import model_config to access model-specific hyperparameters from config.py.
from config import ModelConfig 

class BERTSentimentClassifier(tf.keras.Model):
    """
    Professional BERT-based sentiment classifier model.
    This class inherits from tf.keras.Model, allowing for a custom, trainable Keras model.
    It encapsulates the BERT base and a custom classification head.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased", # Name of the pre-trained BERT model to load (e.g., 'bert-base-uncased').
                 num_classes: int = 2,                 # Number of output sentiment classes (e.g., 2 for positive/negative).
                 dropout_rate: float = 0.1,             # Dropout rate for regularization in the classification layers.
                 max_length: int = 128,                 # Maximum sequence length for input tokens.
                 **kwargs):                             # Allows passing additional keyword arguments to the base class.
        super().__init__() # Initialize the base Keras Model class. This line was fixed in Trainer.py, ensure consistent fix here too.
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        
        # Initialize the tokenizer corresponding to the pre-trained BERT model.
        # This tokenizer converts text into numerical input IDs and attention masks.
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Load the pre-trained TFBertModel (TensorFlow version of BERT).
        # This forms the powerful backbone of our sentiment classifier.
        self.bert = TFBertModel.from_pretrained(model_name)
        # Define a Dropout layer to prevent overfitting during training.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Define the final classification layer.
        # `softmax` activation outputs probabilities for each class, summing to 1.
        self.classifier = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax', 
            name='classifier'     # Assign a name to the layer for better visualization/debugging.
        )
        
        logger.info(f"Initialized BERT model: {model_name}")
    
    def tokenize_texts(self, texts):
        """
        Tokenizes input texts using the pre-trained BERT tokenizer.
        This prepares raw text for input into the BERT model.
        
        Args:
            texts: A list of text strings to be tokenized.
            
        Returns:
            A dictionary containing tokenized inputs as TensorFlow tensors
            (input_ids, attention_mask, etc.).
        """
        return self.tokenizer(
            texts,
            padding=True,              # Pads sequences to the `max_length` or the longest in the batch.
            truncation=True,           # Truncates sequences longer than `max_length`.
            max_length=self.max_length, # Uses the `max_length` defined in model configuration.
            return_tensors='tf'        # Ensures the output is in TensorFlow tensor format.
        )
    
    def call(self, inputs, training=False):
        """
        Defines the forward pass logic of the model.
        This method is called when the model is executed (e.g., during training or prediction).
        
        Args:
            inputs: A dictionary of tokenized inputs (typically 'input_ids' and 'attention_mask').
            training (bool): A boolean indicating whether the model is currently in training mode.
                             This is used to control the behavior of layers like Dropout (active during training).
        
        Returns:
            tf.Tensor: The output logits (raw scores) or probabilities from the classification layer.
        """
        # Pass the input IDs and attention mask through the BERT model.
        # `training` argument is passed to control BERT's internal dropout layers.
        bert_outputs = self.bert(inputs, training=training)
        
        # Extract the pooled output. For classification tasks, this typically represents
        # the aggregated information of the entire sequence, usually from the [CLS] token.
        pooled_output = bert_outputs.pooler_output
        
        # Apply the dropout layer. It is only active when `training` is True.
        pooled_output = self.dropout(pooled_output, training=training)
        
        # Pass the pooled output through the final classification layer.
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_config(self):
        """
        Returns the model's configuration parameters.
        This method is required for Keras to correctly serialize and deserialize the model.
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'max_length': self.max_length
        }

class BERTModelBuilder:
    """
    A static helper class for building BERT-based models.
    It provides a clean interface to construct model instances using the Keras Functional API.
    """
    
    @staticmethod
    def build_functional_model(model_config: ModelConfig) -> tf.keras.Model:
        """
        Builds a BERT-based sentiment classification model using the Keras Functional API.
        The Functional API is preferred for its flexibility in defining complex architectures.
        
        Args:
            model_config (ModelConfig): A configuration object containing hyperparameters
                                        like model name, max length, number of classes, and dropout rate.
                                        
        Returns:
            tf.keras.Model: A compiled Keras model ready for training or prediction.
        """
        
        # Define the input layers for the BERT model: input_ids and attention_mask.
        # `input_ids` are the numerical representations of tokens.
        input_ids = tf.keras.layers.Input(
            shape=(model_config.max_length,), # Input shape is (sequence_length,), batch size is implicit (None).
            dtype=tf.int32,                   # Data type for token IDs is integer.
            name='input_ids'                  # A name for the input layer, useful for model summaries and debugging.
        )
        # `attention_mask` indicates which tokens are real and which are padding, crucial for BERT.
        attention_mask = tf.keras.layers.Input(
            shape=(model_config.max_length,), 
            dtype=tf.int32, 
            name='attention_mask'
        )
        
        # Load the pre-trained TensorFlow BERT model from Hugging Face.
        # This is the backbone of our classification model.
        bert = TFBertModel.from_pretrained(model_config.model_name)
        # Pass the input layers through the BERT model.
        bert_outputs = bert(input_ids, attention_mask=attention_mask)
        
        # Extract the pooled output from BERT. This is typically the representation of the [CLS] token,
        # which is used as the aggregate representation of the entire input sequence for classification.
        pooled_output = bert_outputs.pooler_output
        
        # Add custom classification layers on top of BERT's output.
        # Dropout layer for regularization to prevent overfitting.
        x = tf.keras.layers.Dropout(model_config.dropout_rate)(pooled_output) 
        # An additional Dense layer with ReLU activation for non-linear transformation.
        x = tf.keras.layers.Dense(128, activation='relu')(x) 
        # Another Dropout layer.
        x = tf.keras.layers.Dropout(model_config.dropout_rate)(x)
        # Final output Dense layer with softmax activation for multi-class probability distribution.
        outputs = tf.keras.layers.Dense(
            model_config.num_classes, 
            activation='softmax' 
        )(x)
        
        # Construct the full Keras Model by specifying its inputs and outputs.
        model = tf.keras.Model(
            inputs=[input_ids, attention_mask], # List of input layers.
            outputs=outputs,                   # Output tensor from the final layer.
            name='bert_sentiment_classifier'   # A descriptive name for the entire model.
        )
        
        return model