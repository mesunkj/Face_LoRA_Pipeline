import os
import shutil
import cv2
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.r_gan_upscaler import RGANUpscaler
from src.preprocessing.auto_tagger import AutoTagger

class DataQC:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.r_gan_upscaler = RGANUpscaler(config.RGAN_MODEL_PATH)
        self.auto_tagger = AutoTagger()

    def process_face_folder(self, face_id, face_folder_path, output_base_dir):
        """
        Process a single folder of face images.
        """
        # Kohya SS expects a subfolder with <repeats>_<concept> format inside the train_data_dir
        base_target_dir = os.path.join(output_base_dir, face_id)
        image_target_dir = os.path.join(base_target_dir, f"10_{face_id}")
        os.makedirs(image_target_dir, exist_ok=True)

        
        valid_images = [f for f in os.listdir(face_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\n[DataQC] Processing face ID: {face_id} ({len(valid_images)} images found)")
        
        processed_count = 0
        used_rgan_count = 0
        
        for idx, img_name in enumerate(valid_images):
            img_path = os.path.join(face_folder_path, img_name)
            base_ext = os.path.splitext(img_name)[1]
            out_img_name = f"{face_id}_{idx:03d}{base_ext}"
            out_img_path = os.path.join(image_target_dir, out_img_name)
            
            # Step 1: Detect and crop face
            cropped_face, (w, h) = self.face_detector.detect_and_crop(img_path)
            
            if cropped_face is None:
                print(f"  [Warning] No face detected in {img_name}, skipping.")
                continue
                
            # Step 2: Check resolution and upscale if needed
            r_gan_used = False
            if w < config.MIN_FACE_RESOLUTION or h < config.MIN_FACE_RESOLUTION:
                if config.ENABLE_RGAN_UPSCALING:
                    # Save temporary crop to upscale
                    temp_crop_path = os.path.join(image_target_dir, f"temp_{out_img_name}")
                    cv2.imwrite(temp_crop_path, cropped_face)
                    
                    # Upscale
                    scale = max(2, int(config.TARGET_FACE_RESOLUTION / min(w, h)) + 1)
                    self.r_gan_upscaler.upscale(temp_crop_path, out_img_path, scale=scale)
                    os.remove(temp_crop_path)
                    
                    # Read the upscaled image for final resizing to exact target resolution
                    upscaled_face = cv2.imread(out_img_path)
                    final_face = cv2.resize(upscaled_face, (config.TARGET_FACE_RESOLUTION, config.TARGET_FACE_RESOLUTION))
                    cv2.imwrite(out_img_path, final_face)
                    r_gan_used = True
                    used_rgan_count += 1
                else:
                    print(f"  [Warning] Resolution too low ({w}x{h}) and R-GAN disabled, skipping {img_name}.")
                    continue
            else:
                # Resize directly to target resolution
                final_face = cv2.resize(cropped_face, (config.TARGET_FACE_RESOLUTION, config.TARGET_FACE_RESOLUTION))
                cv2.imwrite(out_img_path, final_face)
                
            # Step 3: Auto-tagging
            self.auto_tagger.create_tag_file(out_img_path, face_id)
            processed_count += 1
            
        return {
            "face_id": face_id,
            "processed_count": processed_count,
            "r_gan_used_count": used_rgan_count,
            "output_dir": base_target_dir
        }

    def run(self):
        """
        Runs the full QC and pre-processing pipeline over all folders in the training data directory.
        """
        print(f"[DataQC] Starting data quality control and pre-processing pipeline...")
        preprocessed_dir = os.path.join(config.OUTPUT_DIR, "preprocessed_data")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        results = []
        if not os.path.exists(config.TRAINING_DATA_DIR):
            print(f"[DataQC] Error: Training data directory not found at {config.TRAINING_DATA_DIR}")
            return results
            
        for item in os.listdir(config.TRAINING_DATA_DIR):
            item_path = os.path.join(config.TRAINING_DATA_DIR, item)
            if os.path.isdir(item_path):
                # Folder name acts as the Face ID (e.g. face_id_001)
                result = self.process_face_folder(item, item_path, preprocessed_dir)
                results.append(result)
                
        print(f"\n[DataQC] Pipeline finished. Processed {len(results)} face IDs.")
        return results

if __name__ == "__main__":
    qc = DataQC()
    qc.run()
