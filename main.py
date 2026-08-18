import os
import config
from src.preprocessing.data_qc import DataQC
from src.training.kohya_runner import KohyaRunner
from src.registry.logger import Logger
from src.evaluation.generator import Generator
from src.evaluation.similarity_scorer import SimilarityScorer

def main():
    print("==================================================")
    print("   Automated Face LoRA Production Line Pipeline   ")
    print("==================================================\n")

    # 1. Initialization
    qc_module = DataQC()
    training_module = KohyaRunner()
    logger_module = Logger()
    eval_generator = Generator()
    eval_scorer = SimilarityScorer()

    # 2. Phase 1: Data QC & Pre-processing
    print(">>> PHASE 1: Data QC & Pre-processing")
    qc_results = qc_module.run()
    
    if not qc_results:
        print("No data processed. Exiting pipeline.")
        return

    # Process each valid Face ID
    for result in qc_results:
        face_id = result['face_id']
        processed_count = result['processed_count']
        dataset_dir = result['output_dir']
        r_gan_used = result['r_gan_used_count'] > 0
        
        if processed_count == 0:
            print(f"Skipping training for {face_id} due to 0 processed images.")
            continue
            
        # 3. Phase 2: Automated Training
        print(f"\n>>> PHASE 2: Training {face_id}")
        lora_path = training_module.train(face_id, dataset_dir)
        
        if not lora_path:
            print(f"Training failed for {face_id}.")
            continue
            
        # 4. Phase 3: Logging (Initial)
        print(f"\n>>> PHASE 3: Logging {face_id}")
        logger_module.log_training(face_id, lora_path, trigger_word=face_id, r_gan_used=r_gan_used)
        
        # 5. Phase 4: Evaluation
        print(f"\n>>> PHASE 4: Evaluating {face_id}")
        eval_output_dir = os.path.join(config.OUTPUT_DIR, "eval", face_id)
        
        # 5.1 Generate test images
        generated_images = eval_generator.generate_test_images(face_id, lora_path, eval_output_dir)
        
        # 5.2 Calculate similarity score against ORIGINAL training images
        original_images_dir = os.path.join(config.TRAINING_DATA_DIR, face_id)
        score = eval_scorer.calculate_score(original_images_dir, generated_images)
        
        # 5.3 Update registry with score
        logger_module.update_evaluation_score(face_id, score)
        
    print("\n==================================================")
    print("   Pipeline Execution Completed!                  ")
    print("==================================================")

if __name__ == "__main__":
    # Ensure base directories exist
    os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Create a dummy folder and image for testing if training_data is empty
    if not os.listdir(config.TRAINING_DATA_DIR):
        print("No training data found. Creating a dummy test directory...")
        dummy_dir = os.path.join(config.TRAINING_DATA_DIR, "test_face_001")
        os.makedirs(dummy_dir, exist_ok=True)
        import cv2
        import numpy as np
        # Create a dummy blank image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dummy_dir, "dummy.jpg"), img)
        print(f"Created dummy data at {dummy_dir}")
        
    main()
