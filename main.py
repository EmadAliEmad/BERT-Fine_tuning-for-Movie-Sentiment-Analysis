%%writefile main.py
import argparse # For command-line argument parsing (not directly used in Notebook but good practice for standalone scripts).
from pathlib import Path # For object-oriented filesystem paths, making path manipulation robust and cross-platform.
import sys # For system-specific parameters and functions, used here to add current directory to Python path.
import json # For working with JSON data, used for saving/loading configuration and history.
from loguru import logger # For professional and highly configurable logging.
from rich.console import Console # From the Rich library, for beautiful and structured console output.
from rich.table import Table # From Rich, for creating formatted tables in the console.
from rich.panel import Panel # From Rich, for creating visually distinct panels in console output.
from rich.progress import track # From Rich, for easily displaying progress bars (used implicitly by Rich when logging).
import tensorflow as tf # Core deep learning library by Google, fundamental for model operations.
from transformers import TFBertModel # Hugging Face's TensorFlow implementation of the BERT model, crucial for loading.

# Import project's custom modules, ensuring modular and organized code.
from config import model_config, training_config, project_config # Project-wide configuration settings.
from logger import setup_logging # Function to configure the logging system.
from data_loader import DataProcessor # Class to handle data loading, cleaning, and splitting.
from model import BERTModelBuilder # Class to build our BERT model architecture.
from trainer import BERTTrainer # Class to encapsulate the model training process.
from evaluator import ModelEvaluator # Class to handle model evaluation and visualization.

console = Console() # Initialize a Rich Console instance for pretty console printing.

def print_project_info():
    """
    Prints a formatted table summarizing the project's key components and their readiness status.
    This provides a quick overview of the project's structure and capabilities using Rich.
    """
    table = Table(title="🚀 Professional BERT Sentiment Analysis")
    table.add_column("Component", style="cyan", no_wrap=True) # Column for component name.
    table.add_column("Status", style="green") # Column for status (e.g., "✅ Ready").
    table.add_column("Description", style="white") # Column for component description.
    
    # Add rows describing each major component of the project.
    table.add_row("Data Processing", "✅ Ready", "Advanced text cleaning and tokenization")
    table.add_row("Model Architecture", "✅ Ready", "BERT-base with custom classification head")
    table.add_row("Training Pipeline", "✅ Ready", "Professional training with callbacks")
    table.add_row("Evaluation Suite", "✅ Ready", "Comprehensive metrics and visualizations")
    table.add_row("Configuration", "✅ Ready", "Modular configuration management")
    table.add_row("Logging", "✅ Ready", "Enhanced logging with Rich and Loguru")
    
    console.print(table) # Display the formatted table to the console.

def main():
    """
    The main execution pipeline for the BERT Sentiment Analysis project.
    It orchestrates data loading, model building/loading, training, evaluation,
    and visualization generation. This function encapsulates the entire workflow.
    """
    
    # Setup the logging system as the very first action to ensure all subsequent messages are logged.
    setup_logging()
    
    # Print the project's introductory information table.
    print_project_info()
    
    try:
        logger.info("Starting BERT Sentiment Analysis Pipeline")
        
        # Initialize the data processor, which handles IMDB dataset operations.
        data_processor = DataProcessor()
        
        # Load and preprocess the IMDB dataset.
        console.print("\n[bold blue]📊 Loading and Processing Data...[/bold blue]")
        # Loads 10,000 samples for efficient training/testing. This number can be adjusted in config.py.
        texts, labels = data_processor.load_imdb_dataset(num_samples=10000) 
        
        # Create the train, validation, and test data splits from the loaded data.
        data_splits = data_processor.create_data_splits(texts, labels)

        # Define the expected path for the saved best model.
        model_path = project_config.models_dir / "best_model.h5"
        
        model = None # Initialize model variable.
        history = None # Initialize history variable; will be populated if model is trained.

        # Check if a trained model already exists on disk.
        if model_path.exists():
            console.print(f"\n[bold green]✅ Found existing model at {model_path}. Loading model...[/bold green]")
            # Load the saved Keras model from the H5 file.
            # `custom_objects={'TFBertModel': TFBertModel}` is crucial here. It tells Keras how to
            # interpret and reconstruct the `TFBertModel` layer, which is a custom layer from Hugging Face
            # and not part of standard Keras. This resolves the `ValueError: Unknown layer: 'TFBertModel'` error.
            model = tf.keras.models.load_model(
                str(model_path), 
                custom_objects={'TFBertModel': TFBertModel}, # Explicitly pass the custom layer.
                compile=False # Do not compile during loading; re-compile explicitly below for evaluation.
            )
            # Re-compile the loaded model. This is necessary to correctly set up the optimizer and metrics
            # for any subsequent evaluation or potential fine-tuning, even if not explicitly trained in this run.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
            console.print("[green]✅ Model loaded and re-compiled successfully for evaluation.[/green]")
        else:
            # If no trained model is found, proceed with building and training.
            console.print("\n[bold blue]🏗️ No existing model found. Building and Training BERT Model...[/bold blue]")
            # Build the BERT model using the predefined builder class and model configuration.
            model = BERTModelBuilder.build_functional_model(model_config)
            
            console.print(f"[green]✅ Model built successfully![/green]")
            # Display the total number of trainable parameters in the model.
            console.print(f"Model parameters: {model.count_params():,}") 
            
            # Initialize the trainer with the newly built model and configurations.
            trainer = BERTTrainer(model, model_config, training_config)
            
            # Start the model training process.
            console.print("\n[bold blue]🎯 Training Model...[/bold blue]")
            history = trainer.train(data_splits['train'], data_splits['validation'])
            
            # Save the training history (metrics over epochs) and model configuration.
            trainer.save_training_artifacts()
            console.print(f"[green]✅ Model trained and saved to {model_path}[/green]")

        # Ensure the `trainer` object is initialized even if the model was loaded (not trained in this run).
        # This is important because the `evaluator` depends on `trainer.tokenizer`.
        if model is not None and 'trainer' not in locals(): # Check if 'trainer' was not created by the 'else' block.
            trainer = BERTTrainer(model, model_config, training_config) # Re-initialize trainer just to get the tokenizer.

        # Evaluate the model on the dedicated test set.
        console.print("\n[bold blue]📈 Evaluating Model...[/bold blue]")
        evaluator = ModelEvaluator(model, trainer.tokenizer) # Pass the loaded model and its tokenizer to the evaluator.
        evaluation_results = evaluator.evaluate_model(data_splits['test'])
        
        console.print(Panel(f"[green]✅ Model evaluation complete![/green]"))
        console.print("\n[bold yellow]Performance Metrics:[/bold yellow]")
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Value", style="bold green")
        for metric, value in evaluation_results['metrics'].items():
            metrics_table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
        console.print(metrics_table)
        
        console.print("\n[bold yellow]Classification Report:[/bold yellow]")
        console.print(json.dumps(evaluation_results['classification_report'], indent=2))
        
        # Generate and save various visualizations of the model's performance.
        console.print("\n[bold blue]📊 Generating Visualizations...[/bold blue]")
        # Plot training history only if the model was actually trained in this run (i.e., `history` object exists).
        if history: 
            evaluator.plot_training_history(history)
            console.print("[green]✅ Training history plot generated.[/green]")
        else:
            console.print("[yellow]Skipping training history plot as model was loaded, not trained.[/yellow]")

        evaluator.plot_confusion_matrix(evaluation_results['confusion_matrix'])
        evaluator.plot_roc_curve(evaluation_results['true_labels'], evaluation_results['predictions'])
        console.print("[green]✅ Confusion Matrix and ROC Curve plots generated.[/green]")
        console.print("[green]✅ All visualizations saved to 'outputs' directory.[/green]")
        
        console.print(Panel("[bold green]✨ Project completed successfully![/bold green]"))

    except Exception as e:
        # Catch any exceptions during the pipeline execution and log them for debugging.
        logger.exception(f"An error occurred during the pipeline execution: {e}")
        console.print(Panel(f"[bold red]❌ An error occurred: {e}[/bold red]", style="red"))

if __name__ == "__main__":
    # Ensures main() is called when the script is executed directly (not imported).
    main()