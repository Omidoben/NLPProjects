# config.py
import os

# Paths
# These paths assume files are uploaded directly to the Colab /content/ directory
DATA_FILE_PATH = "complaints.csv" 

MODELS_DIR = "./models" # This directory will be created in /content/
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "complaints_classifier")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
LOGS_DIR = os.path.join(MODELS_DIR, "logs")


# Data Columns
TEXT_COLUMN = "consumer_complaint_narrative" 
LABEL_COLUMN = "product" 

# Model Configuration
MODEL_NAME = "distilbert-base-uncased" # Or "bert-base-uncased", "roberta-base", etc.
MAX_SEQUENCE_LENGTH = 256 # Max tokens for input
TEST_SIZE = 0.2 
RANDOM_SEED = 42

# Training Hyperparameters
NUM_TRAIN_EPOCHS = 5
BATCH_SIZE_TRAIN = 8 
BATCH_SIZE_EVAL = 16 
LEARNING_RATE = 5e-5 
WEIGHT_DECAY = 0.01 
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50


# Weight decay (also known as L2 regularization) is a technique used to prevent overfitting. It adds a penalty to the loss function that is proportional to the square of the magnitude of the model's weights. This discourages the model from assigning very large weights to specific features, leading to a more generalized model.