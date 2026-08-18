import os

class AutoTagger:
    def __init__(self):
        pass
        
    def create_tag_file(self, image_path, trigger_word):
        """
        Creates a text file alongside the image containing the trigger word 
        and optionally some base tags (e.g., '1girl, face' or '1boy, face').
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        base_name = os.path.splitext(image_path)[0]
        txt_path = f"{base_name}.txt"
        
        # Write the trigger word to the text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{trigger_word}, a photo of a face, highly detailed")
            
        print(f"[AutoTagger] Created tag file: {txt_path} with trigger: {trigger_word}")
        return txt_path
