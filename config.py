import os
import sys

# Detect if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

# Base paths
if IN_COLAB:
    # Google Drive default working directory
    BASE_DIR = "/content/drive/MyDrive/app/AI/Lora/Face_LoRA_Pipeline"
else:
    # Local working directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
REGISTRY_FILE = os.path.join(BASE_DIR, "lora_registry.json")
MEMO_FILE = os.path.join(BASE_DIR, "LORA_MEMO.md")

# Pre-processing Configuration
MIN_FACE_RESOLUTION = 512
TARGET_FACE_RESOLUTION = 512
ENABLE_RGAN_UPSCALING = True
# R-GAN specific parameters (e.g., path to executable or model)
RGAN_MODEL_PATH = "realesrgan-x4plus.pth"

# Training Configuration (Kohya_ss)
KOHYA_DIR = os.path.join(BASE_DIR, "kohya_ss")  # Adjust this if Kohya is installed elsewhere
TRAIN_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 1000
SAVE_EVERY_N_EPOCHS = 1
NETWORK_DIM = 64
NETWORK_ALPHA = 32

# Evaluation Configuration
EVAL_PROMPT_TEMPLATE = "A portrait of {trigger_word}, raw photo, highly detailed, 8k uhd, dslr"
EVAL_NEGATIVE_PROMPT = "blurry, out of focus, disfigured, low quality, bad anatomy"
NUM_TEST_IMAGES = 10
EVAL_MODEL_CHECKPOINT = "v1-5-pruned-emaonly.safetensors" # Base model for evaluation generation
SIMILARITY_THRESHOLD = 0.50 # 50% threshold for passing (InsightFace cosine similarity)
