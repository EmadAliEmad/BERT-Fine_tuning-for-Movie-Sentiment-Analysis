# %%writefile data_loader.py
import pandas as pd # Library for data manipulation (e.g., concatenating arrays, not directly used for DataFrame operations here).
import numpy as np # Fundamental package for numerical computing, used for array operations like concatenation.
import tensorflow as tf # Core deep learning library, specifically used here for loading the IMDB dataset from Keras.
from sklearn.model_selection import train_test_split # Utility for splitting datasets into random train and test subsets.
from sklearn.utils import shuffle # Utility for randomizing the order of elements in lists.
from typing import Tuple, List, Dict, Optional # Used for type hints, improving code readability and maintainability.
import re # Module for regular expressions, used for pattern matching and text manipulation (cleaning).
import html # Module for working with HTML entities, used for decoding HTML in text cleaning.
from loguru import logger # Used for structured and informative logging of data processing steps.

class DataProcessor:
    """
    Advanced data processing pipeline for sentiment analysis.
    This class handles loading the raw IMDB dataset, cleaning text content,
    and splitting the data into training, validation, and test sets.
    """

    def __init__(self):
        """
        Initializes the DataProcessor with mappings for sentiment labels.
        This provides a clear way to convert between string labels and numerical IDs.
        """
        self.label_mapping = {"negative": 0, "positive": 1} # Maps string labels to numerical IDs.
        self.reverse_label_mapping = {0: "negative", 1: "positive"} # Maps numerical IDs back to string labels.

    def clean_text(self, text: str) -> str:
        """
        Applies an advanced text cleaning pipeline to a single text string.
        This function is crucial for standardizing input text before tokenization,
        ensuring consistency with how the model was trained.

        Args:
            text (str): The input text string to be cleaned.

        Returns:
            str: The cleaned text string.
        """
        # Ensure the input is a string; return empty if not to prevent errors.
        if not isinstance(text, str):
            return ""

        # Decode HTML entities (e.g., convert '&' to '&').
        # This prevents HTML encoding from interfering with sentiment analysis.
        text = html.unescape(text)

        # Remove HTML tags (e.g., '<br />' becomes '').
        # HTML tags are noise for sentiment analysis.
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace: replace multiple spaces, tabs, and newlines with a single space.
        # This ensures consistent spacing and removes extra blank lines.
        text = re.sub(r'\s+', ' ', text)

        # Remove any characters that are not alphanumeric, whitespace, or common punctuation.
        # This helps in removing special symbols or emojis that might not be in BERT's vocabulary.
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)

        # Convert the entire text to lowercase and remove any leading/trailing whitespace.
        # Lowercasing helps standardize words (e.g., "Good" and "good" are treated the same).
        text = text.lower().strip()

        return text

    def load_imdb_dataset(self, num_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """
        Loads the IMDB movie review dataset directly from TensorFlow Keras datasets.
        It then decodes the numerical review sequences back into human-readable text,
        applies the defined cleaning process, and optionally limits the number of samples.

        Args:
            num_samples (Optional[int]): If provided, only this many samples will be loaded
                                         from the full dataset. Useful for faster debugging
                                         and initial development.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing:
                - List of cleaned text reviews.
                - List of corresponding numerical sentiment labels (0 for negative, 1 for positive).
        """
        logger.info("Loading IMDB dataset...")

        try:
            # Load the IMDB dataset. This includes numerical sequences of reviews and their labels.
            # The data is downloaded the first time this function is called.
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

            # Retrieve the word index mapping from the IMDB dataset.
            # This dictionary maps words to integer IDs.
            word_index = tf.keras.datasets.imdb.get_word_index()
            # Create a reverse mapping from integer IDs back to words.
            # The '-3' accounts for special tokens (padding, start-of-sequence, unknown).
            reverse_word_index = {value: key for key, value in word_index.items()}

            # Nested helper function to decode a single numerical review sequence back to text.
            def decode_review(encoded_review):
                return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

            # Convert all numerical training and testing reviews into text strings.
            x_train_text = [decode_review(review) for review in x_train]
            x_test_text = [decode_review(review) for review in x_test]

            # Combine all texts and labels into single lists for unified processing.
            all_texts = x_train_text + x_test_text
            all_labels = np.concatenate([y_train, y_test])

            # Apply the text cleaning function to all combined texts.
            logger.info("Cleaning text data...")
            cleaned_texts = [self.clean_text(text) for text in all_texts]

            # Filter out reviews that became too short or empty after cleaning.
            # This prevents very short, uninformative texts from being processed.
            valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 10]
            cleaned_texts = [cleaned_texts[i] for i in valid_indices]
            all_labels = [all_labels[i] for i in valid_indices]

            # Optionally limit the number of samples for faster experimentation.
            if num_samples and num_samples < len(cleaned_texts):
                # Shuffle the data before slicing to ensure randomness in selected samples.
                cleaned_texts, all_labels = shuffle(cleaned_texts, all_labels, random_state=42)
                cleaned_texts = cleaned_texts[:num_samples] # Select first `num_samples`.
                all_labels = all_labels[:num_samples]

            logger.info(f"Loaded {len(cleaned_texts)} samples")
            return cleaned_texts, all_labels

        except Exception as e:
            # Log any errors that occur during dataset loading and re-raise them.
            logger.error(f"Error loading IMDB dataset: {e}")
            raise

    def create_data_splits(self, texts: List[str], labels: List[int],
                          test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """
        Splits the provided texts and labels into distinct training, validation, and test datasets.
        It uses a stratified splitting approach to ensure that the class distribution (positive/negative)
        is preserved in each subset.

        Args:
            texts (List[str]): A list of cleaned text samples.
            labels (List[int]): A list of corresponding numerical sentiment labels.
            test_size (float): The proportion of the dataset to allocate to the final test set (e.g., 0.2 for 20%).
            val_size (float): The proportion of the remaining data (after test split) to allocate to the validation set.

        Returns:
            Dict: A dictionary containing three keys ('train', 'validation', 'test'),
                  each mapping to another dictionary with 'texts' and 'labels' for that split.
        """
        logger.info("Creating data splits...")

        # First split: Separate out the final test set.
        # `stratify=labels` ensures that the proportion of classes (0s and 1s) is the same in both X_temp and X_test.
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Second split: Divide the remaining data (X_temp, y_temp) into training and validation sets.
        # The validation size is adjusted because it's a split of the *remaining* data, not the original full dataset.
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        # Organize the split datasets into a dictionary for easy access.
        data_splits = {
            'train': {'texts': X_train, 'labels': y_train},
            'validation': {'texts': X_val, 'labels': y_val},
            'test': {'texts': X_test, 'labels': y_test}
        }

        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return data_splits

    def create_tf_dataset(self, texts: List[str], labels: List[int],
                         batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Converts Python lists of text samples and their corresponding labels into a
        highly performant TensorFlow `tf.data.Dataset` object. This format is
        optimized for feeding data to TensorFlow models during training.

        Args:
            texts (List[str]): A list of text samples.
            labels (List[int]): A list of corresponding numerical labels.
            batch_size (int): The number of elements (samples) to include in each batch
                              of the dataset.
            shuffle (bool): If True, the dataset will be shuffled. Recommended for training data.

        Returns:
            tf.data.Dataset: A TensorFlow dataset, which is batched and prefetched
                             for efficient data loading and processing during model training.
        """
        # Create a tf.data.Dataset from slices of the input texts and labels.
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

        # Shuffle the dataset if specified.
        # `buffer_size` is used for shuffling, ensuring efficient shuffling.
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)

        # Batch the dataset into smaller chunks for efficient processing by the model.
        dataset = dataset.batch(batch_size)
        # Use prefetching to allow the data pipeline to prepare batches in the background
        # while the model is processing the current batch, improving performance.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset