import os
import shutil

# Define where your raw images are currently
SOURCE_DIR = 'pcb-defect-dataset/train/images'
# Define where the new structured folder will be
TARGET_DIR = 'pcb-defect-dataset/train_structured'

# The 6 classes required for the VisionSpec QC system
CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

def organize_dataset():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    for cls in CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    count = 0

    for filename in files:
        # Match filename to class (e.g., "01_short_01.jpg" goes to "Short")
        for cls in CLASSES:
            if cls.lower() in filename.lower():
                shutil.copy(os.path.join(SOURCE_DIR, filename), 
                            os.path.join(TARGET_DIR, cls, filename))
                count += 1
                break
    
    print(f"Done! Successfully organized {count} images into 6 class folders.")

if __name__ == "__main__":
    organize_dataset()