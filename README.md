# Plant Disease Classification using Transfer Learning

## Overview
This project implements and compares different deep learning approaches for plant disease classification. Using transfer learning with pre-trained convolutional neural networks (CNNs), we explore both feature extraction and fine-tuning techniques on the New Plant Diseases Dataset. The project demonstrates how to effectively utilize powerful CNN architectures to achieve high accuracy in identifying plant diseases, which can help in early detection and treatment in agricultural applications.

## Dataset
We used the [New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset) from Kaggle, which contains:
- 38 classes of plant diseases and healthy plants
- ~1,600-2,000 images per class
- Training, validation, and test splits

The dataset includes various plant types such as apple, cherry, corn, grape, peach, potato, and tomato with different disease conditions.

## Methodology
We implemented and compared two different CNN architectures:

### 1. ResNet101
- **Depth**: 101 layers
- **Parameters**: ~44.7M
- **Design Philosophy**: Uses residual connections to solve the vanishing gradient problem
- **Feature Extraction**: All pre-trained layers frozen, only the classification head is trained
- **Fine-Tuning**: Top ~5% of layers unfrozen and retrained with a lower learning rate

### 2. EfficientNetB0
- **Parameters**: ~5.3M (significantly smaller than ResNet101)
- **Design Philosophy**: Compound scaling for better efficiency-accuracy trade-off
- **Feature Extraction**: All pre-trained layers frozen, only the classification head is trained
- **Fine-Tuning**: Top layers unfrozen and retrained with a lower learning rate

## Results

### Performance Metrics

| Model | Approach | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|----------|-------------------|---------------------|---------------|-----------------|
| ResNet101 | Feature Extraction | 97.28% | 97.34% | 0.0906 | 0.0883 |
| ResNet101 | Fine-Tuning | 98.79% | 98.77% | 0.0453 | 0.0446 |
| EfficientNetB0 | Feature Extraction | 96.47% | 96.75% | 0.1323 | 0.1269 |
| EfficientNetB0 | Fine-Tuning | 98.30% | 98.49% | 0.0588 | 0.0555 |

### Key Findings
- Fine-tuning consistently outperforms feature extraction for both architectures
- ResNet101 achieves slightly higher accuracy than EfficientNetB0
- All models show excellent generalization with minimal overfitting
- ResNet101 achieves lower loss values, suggesting more confident predictions
- EfficientNetB0 offers significantly reduced parameter count and faster inference time

## Visualizations
The project includes various visualizations:
- Class distribution analysis
- Training and validation curves for all models
- Confusion matrices to identify misclassifications
- Grad-CAM visualizations to see which regions the models focus on when making predictions

## Grad-CAM Analysis
We used Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which parts of the plant images the models focus on when making predictions. This helps in understanding model decision-making and confirms that the models are correctly identifying disease-relevant regions rather than background elements.

## Conclusion
Our analysis shows that:

1. Both ResNet101 and EfficientNetB0 achieve excellent performance on plant disease classification, with fine-tuning providing significant improvements over feature extraction.

2. While ResNet101 achieves slightly higher accuracy, EfficientNetB0 offers a much better efficiency-accuracy trade-off with 8x fewer parameters.

3. The high accuracy achieved (98.77% for ResNet101 fine-tuning) demonstrates the effectiveness of transfer learning for specialized image classification tasks like plant disease detection.

4. Grad-CAM visualizations confirm that the models correctly focus on disease-relevant regions when making classifications.

## Potential Improvements
- Model ensemble combining predictions from both architectures
- Additional data augmentation specific to plant disease variance
- Progressive unfreezing during fine-tuning
- Hyperparameter optimization for further performance gains
- Testing newer architectures like EfficientNetV2 or Vision Transformers

## Authors
- Ali Abdullah (413814) - Fine-tuning, visualization, and comparative analysis
- Rabbiya Riaz (406636) - Dataset processing and feature extraction implementation

## Requirements
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Kaggle API (for dataset download)

## Usage
The implementation is available as a Jupyter notebook that can be run in Google Colab.

```python
# Example of loading and using the trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the fine-tuned model
model = load_model('resnet101_fineTune.h5')

# Load and preprocess an image
img = image.load_img('plant_image.jpg', target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Make prediction
predictions = model.predict(x)
predicted_class = np.argmax(predictions, axis=1)[0]
print(f"Predicted class index: {predicted_class}")
```
