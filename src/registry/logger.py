import os
import json
import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class Logger:
    def __init__(self):
        self.registry_file = config.REGISTRY_FILE
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump({"faces": []}, f, indent=4)

    def log_training(self, face_id, lora_file, trigger_word, r_gan_used):
        """
        Log the completed training to the JSON registry.
        """
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        # Check if the face_id already exists and update it, else append
        existing_idx = next((i for i, item in enumerate(registry["faces"]) if item["face_id"] == face_id), None)
        
        entry = {
            "name": face_id.replace("_", " ").title(),  # Dummy generation
            "face_id": face_id,
            "lora_file": os.path.basename(lora_file) if lora_file else None,
            "trigger_word": trigger_word,
            "r_gan_used": r_gan_used,
            "training_date": datetime.date.today().isoformat(),
            "evaluation_score": None  # To be filled by evaluation phase
        }
        
        if existing_idx is not None:
            registry["faces"][existing_idx] = entry
        else:
            registry["faces"].append(entry)
            
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=4)
            
        print(f"[Logger] Logged training results for {face_id} to registry.")

    def update_evaluation_score(self, face_id, score):
        """
        Update the evaluation score in the registry.
        """
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        for entry in registry["faces"]:
            if entry["face_id"] == face_id:
                entry["evaluation_score"] = score
                break
                
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=4)
            
        print(f"[Logger] Updated evaluation score for {face_id} to {score:.1f}%")

if __name__ == "__main__":
    logger = Logger()
    logger.log_training("face_id_001", "face_id_001.safetensors", "face_id_001", True)
    logger.update_evaluation_score("face_id_001", 94.2)
