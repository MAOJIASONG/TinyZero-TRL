import colorlog
import transformers
import datasets
import sys
import os


def setup_logger(training_args: transformers.TrainingArguments):
    """
    Setup the logger for the training process.
    """
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
            }
    log_level = training_args.get_process_log_level()
    colorlog.basicConfig(
        log_colors=log_colors, 
        format=fmt_string, 
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level = log_level
    )

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger = colorlog.getLogger(__name__)
    
    # Configure wandb
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    return logger


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project