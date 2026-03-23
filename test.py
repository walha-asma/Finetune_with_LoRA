import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Test on one of your generated images
img = Image.open("results/generated_images/lora_flux2klein_rank8/test/05_A_hand_holding_a_smart_phone_with_the_te.png")
results = reader.readtext(np.array(img), detail=1)

for bbox, text, confidence in results:
    print(f"  conf={confidence:.2f}  text='{text}'")