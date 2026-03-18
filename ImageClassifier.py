# ===============================
# 1. Import Libraries
# ===============================
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 2. Load Pre-trained VGG16 Model
# ===============================
model = VGG16(weights='imagenet')

# ===============================
# 3. List of Images to Analyze
# ===============================
image_paths = [
    r'C:\Users\ehanso13\CobberLearnChemProjects\ImageClassifier\cat.jpg',
    r'C:\Users\ehanso13\CobberLearnChemProjects\ImageClassifier\dog.jpg',
    r'C:\Users\ehanso13\CobberLearnChemProjects\ImageClassifier\horse.jpg'
]

# ===============================
# 4. Process Each Image
# ===============================
for img_path in image_paths:
    # Load and resize image
    img = image.load_img(img_path, target_size=(224, 224))

    # Display image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Convert to numpy array and preprocess
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)

    # Decode top 5 predictions
    decoded_preds = decode_predictions(preds, top=5)[0]

    # Print results
    print(f"Image: {os.path.basename(img_path)}")
    print("Top 5 Predictions:")
    for i, (_, label, prob) in enumerate(decoded_preds):
        print(f"{i + 1}. {label}: {prob * 100:.2f}%")
    print("\n" + "=" * 40 + "\n")

# ===============================
# 5. Reflection / Analysis
# ===============================
# - Compare predictions across all images to check consistency.
# - High-confidence predictions indicate strong model certainty.
# - Lower confidence or inconsistent top predictions reflect ambiguity, similar to batch analysis in chemistry experiments.
# - In chemistry, analyzing multiple samples helps identify trends and ensure reliability; here, multiple images reveal how consistent VGG16 is across related objects.

"""
1. VGG16 vs Chemists Identifying Molecules:
   - VGG16 analyzes images layer by layer, detecting edges, shapes, patterns, and objects.
   - Similarly, chemists examine molecular features (functional groups, bond angles, spectra) to identify compounds.
   - Both rely on recognizing **patterns in structured data** to make informed identifications.

2. Limitations of VGG16:
   - Pre-trained on ImageNet → recognizes common everyday objects, not specialized lab images.
   - May fail on:
       * Microscopic images, chemical structures, or unusual lab equipment
       * Objects not in training set
       * Images with unusual angles, lighting, or obstructions
   - Output can be uncertain if image is ambiguous or very different from training data.

3. Applications to Chemistry:
   - Quality control: automatically classify lab samples, detect mislabeled compounds, or flag defects.
   - Research: categorize molecular models, analyze spectroscopic images, or monitor experiments.
   - Could help save time and reduce human error in repetitive tasks.

4. Ethical Considerations:
   - Reliability: automated systems may make mistakes, so human oversight is essential.
   - Transparency: results should be explainable, so users should know the system’s confidence and limitations.
   - Bias: model trained on non-lab images may misclassify specialized scientific data.
   - Accountability: errors in quality control or research could have safety implications,so responsibility must be clearly defined.


SAMPLE OUTPUTS:

Image: cat.jpg
Top 5 Predictions:
1. tiger_cat: 23.99%
2. lynx: 22.20%
3. tabby: 14.73%
4. Egyptian_cat: 4.73%
5. coyote: 4.57%

========================================

Image: dog.jpg
Top 5 Predictions:
1. Bernese_mountain_dog: 56.01%
2. Appenzeller: 20.61%
3. EntleBucher: 11.91%
4. Greater_Swiss_Mountain_dog: 6.20%
5. Gordon_setter: 1.22%

========================================

Image: horse.jpg
Top 5 Predictions:
1. sorrel: 91.91%
2. worm_fence: 1.40%
3. whippet: 1.04%
4. hartebeest: 0.71%
5. Saluki: 0.54%

========================================

"""
# End of code