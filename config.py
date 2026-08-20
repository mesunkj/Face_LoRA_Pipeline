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
if IN_COLAB:
    KOHYA_DIR = "/content/kohya_ss"
else:
    KOHYA_DIR = os.path.join(BASE_DIR, "kohya_ss")
TRAIN_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 1000
SAVE_EVERY_N_EPOCHS = 1
NETWORK_DIM = 64
NETWORK_ALPHA = 32

# Evaluation Configuration
EVAL_PROMPTS = {
    "scene_standard": "A portrait of {trigger_word}, raw photo, highly detailed, 8k uhd, dslr",
    "scene_close_up": "A macro close-up of {trigger_word}'s face, highly detailed, 8k uhd, dslr",
    "scene_wide": "A wide angle shot of {trigger_word}, full body, highly detailed, 8k uhd, dslr",
    "scene_low_light": "A portrait of {trigger_word} in low light neon cyberpunk, highly detailed, 8k uhd",
    "scene_profile": "A side profile portrait of {trigger_word}, highly detailed, 8k uhd, dslr"
}
EVAL_NEGATIVE_PROMPT = "blurry, out of focus, disfigured, low quality, bad anatomy"
NUM_TEST_IMAGES = 10
EVAL_MODEL_CHECKPOINT = "runwayml/stable-diffusion-v1-5"
SIMILARITY_THRESHOLD = 0.50
FIXED_SEED = 42
EVAL_OUTPUT_DIR_TEMPLATE = os.path.join(OUTPUT_DIR, "eval", "{face_id}_standalone")
