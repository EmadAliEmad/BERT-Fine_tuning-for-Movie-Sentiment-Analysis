#%%writefile config.py
import os # Module for interacting with the operating system (e.g., path operations).
from pathlib import Path # Object-oriented filesystem paths, preferred for cross-platform path handling.
from dataclasses import dataclass # Decorator to easily create classes that store data (config settings).
from typing import Optional # For type hinting, indicating a value can be None or a specific type.

@dataclass
class ModelConfig:
    """Model configuration settings, defining BERT model's core hyperparameters and properties."""
    model_name: str = "bert-base-uncased" # Name of the pre-trained BERT model from Hugging Face Transformers.
    max_length: int = 128               # Maximum sequence length for tokenization; sequences longer are truncated, shorter are padded.
    num_classes: int = 2                # Number of output classes for sentiment (e.g., 2 for positive/negative).
    dropout_rate: float = 0.1           # Dropout probability applied in the classification head for regularization.
    learning_rate: float = 2e-5         # Initial learning rate for the Adam optimizer during training.
    batch_size: int = 16                # Number of samples processed in one forward/backward pass during training.
    epochs: int = 3                     # Number of full passes through the entire training dataset.
    validation_split: float = 0.2       # Proportion of the training data to be reserved for validation.

@dataclass
class TrainingConfig:
    """Training configuration settings, controlling the training process behavior and callback strategies."""
    save_strategy: str = "epoch"        # Defines when to save model checkpoints (e.g., 'epoch' to save after each epoch).
    evaluation_strategy: str = "epoch"  # Defines when to evaluate the model on the validation set (e.g., 'epoch').
    logging_steps: int = 100            # How many steps between logging training progress updates.
    save_total_limit: int = 3           # Maximum number of model checkpoints to keep. Older ones are deleted.
    load_best_model_at_end: bool = True # If True, the model with the best validation metric is loaded at the end of training.
    metric_for_best_model: str = "eval_accuracy" # The metric to monitor to determine the "best" model checkpoint.
    greater_is_better: bool = True      # If True, a higher value for `metric_for_best_model` indicates a better model.

@dataclass
class ProjectConfig:
    """Project paths and settings, defining the directory structure for inputs and outputs."""
    # Project root directory. Critically set to '/kaggle/working/' for execution within Kaggle Notebooks.
    project_root: Path = Path("/kaggle/working").absolute() 
    data_dir: Path = project_root / "data"     # Directory for raw and processed data files.
    models_dir: Path = project_root / "models" # Directory to save trained model checkpoints (e.g., best_model.h5).
    logs_dir: Path = project_root / "logs"     # Directory for application logs (e.g., app.log).
    output_dir: Path = project_root / "outputs" # Directory for evaluation results, reports, and plots.
    temp_dir: Path = project_root / "temp"     # Directory for temporary files created during execution.

    def __post_init__(self):
        """
        Special method that runs automatically after an instance of ProjectConfig is created.
        It ensures all specified project directories exist, creating them if necessary.
        """
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir,
                        self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True) # `parents=True` creates any missing parent directories.

# Initialize configurations: Create single instances of each config class.
# These instances hold the definitive settings and are imported across other modules.
model_config = ModelConfig()
training_config = TrainingConfig()
project_config = ProjectConfig()