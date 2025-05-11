# Alzheimer's Disease MRI Classification with CNNs
This project explores the effectiveness of **Convolutional Neural Networks (CNNs)** for classifying MRI brain scans into four stages of Alzheimer's Disease:  
- Alzheimer’s Disease (AD)
- Early Mild Cognitive Impairment (EMCI)
- Late Mild Cognitive Impairment (LMCI)
- Cognitively Normal (CN)

> Developed as part of the final project for *COSC 423: Machine Learning*, by Lucas Mitchell.

## Goals:
- Evaluate three different CNNs performance on MRI image data
    - ResNet50 (no pretrained weights)
    - Custom CNN
    - LeNet-5 (ReLU activation functions)
- Compare performance of these architectures
- Address overfitting and model robustness

## Dataset
- **Source**: [ADNI-4C MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/abdullahtauseef2003/adni-4c-alzheimers-mri-classification-dataset/data)
- **Classes**:
  - `0` – Alzheimer’s Disease (AD)
  - `1` – Early Mild Cognitive Impairment (EMCI)
  - `2` – Late Mild Cognitive Impairment (LMCI)
  - `3` – Cognitively Normal (CN)
- NOTE: Data Augmentation (particularly random zooming, flipping, and contrast) was inteded to be added, however, due to time constraints with the training it was not implemented.

## Preprocessing & Training Methodology
- **Preprocessing**: Grayscale conversion, re-scaling to 128x128, pixel value normalization
- **Splitting**: 80/20 train-test, with 80/20 train-validation
- **Evaluation Metrics**: Accuracy, F1-score, Confusion Matrix
- **Hyperparameter Tuning**: Dropout rate, pooling type, kernel size, flattening method

## Models Implemented
### 1. **LeNet-5**
- Modified to use ReLU activations
- Struggled with complex features in MRI data

### 2. **ResNet-50**
- Deep (50 layers) network (no pretrained weights)
- Overfit quickly due to limited data with no augmentation

### 3. **Custom CNN (Best Model)**
- 3 Convolutional layers (64, 128, 256 filters, 5x5 kernel)
- ReLU activations + MaxPooling
- Global Average Pooling 
- Dense layers (128, 64 neurons) each with Dropout (rate of 0.5)
- Final dense softmax layer (4 outputs)
- *See the model summary in .ipynb file for more details*

## Results & Performance Comparison
| Model      | Accuracy | F1-Score |
|------------|----------|----------|
| Custom CNN | **0.916**   | **0.922**   |
| ResNet50   | 0.887    | 0.894    |
| LeNet-5    | 0.850    | 0.858    |

- All models performed best on the *Cognitively Normal* class
- Custom CNN showed **least overfitting** and **highest accuracy**
- *See Confusion Matricies and Learning Curves in .ipynb file for more details*

## Limitations and Future Work
- No **data augmentation** applied (could improve generalization)
- Models struggled with differences between disease stages (particularly late-mild and early-mild)
- **Future work**:
  - Incorporate data augmentation (rotation, zoom, contrast)
  - Test with pretrained models (transfer learning)
  - Try more models (AlexNet, EfficientNet, DenseNet)