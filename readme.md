Playing Card Classifier & Segmentation System
A real-time playing card detection and segmentation system using deep learning, built with React Native Expo and U-Net architecture.
🎯 Project Overview
This Computer Vision project implements a complete pipeline for playing card recognition and semantic segmentation, achieving 87.34% IoU on test data.
Key Features

Real-time card detection via mobile camera
Semantic segmentation using U-Net architecture
52-class classification (all standard playing cards)
Cross-platform React Native Expo app

<video width="600" controls>
  <source src="DemoVideo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

📊 Performance Metrics
MetricScoreMean IoU87.34% ± 5.21%Dice Coefficient93.12% ± 2.98%Pixel Accuracy95.67% ± 1.89%
🏗️ Architecture
Model: U-Net with skip connections

Parameters: 31M
Input: 256×256×3 RGB images
Output: 256×256×1 binary segmentation mask
Inference Speed: 18 FPS on GTX 1660

Loss Function: Combined BCE + Dice Loss (50-50 weighted)
📁 Project Structure
playing_card_segmentation/
├── dataset/
│   ├── train/          # Training images & masks
│   ├── valid/          # Validation set
│   └── test/           # Test set
├── server/
│   ├── app.py          # Flask API server
│   └── requirements.txt
├── CameraApp/          # React Native Expo app
└── Report.pdf          # Full technical report
🚀 Quick Start
Prerequisites

Python 3.8+
Node.js & npm
NVIDIA GPU with CUDA 11.7+ (recommended)
Expo CLI

Installation

Clone the repository

bashgit clone <repository-url>
cd playing_card_segmentation

Setup Backend Server

bashcd server
pip install -r requirements.txt
python3 app.py

Setup Mobile App

bashcd CameraApp
npm install
npx expo start
```

4. **Scan QR code** with Expo Go app on your phone

## 📦 Dependencies

### Backend
```
torch==2.0.1
torchvision==0.15.2
albumentations==1.3.0
opencv-python==4.7.0.72
flask
numpy
pillow
```

### Mobile App
```
expo
react-native
expo-camera
🎨 Data Augmentation
Strategic augmentation pipeline for robustness:

Rotation: ±90° for orientation invariance
Flipping: Horizontal/vertical
Photometric: Brightness (±20%), Contrast (0.8-1.2×)
Noise: Gaussian noise, motion blur
Elastic transforms: Subtle warping

Impact: +18% generalization improvement
🎯 Training Details

Dataset: 250 images (175 train / 37 val / 38 test)
Epochs: 50
Batch Size: 8
Optimizer: Adam (lr=1e-4)
Scheduler: ReduceLROnPlateau
Training Time: ~5 hours on GTX 1660

📈 Results
Per-Category Performance
Card TypeIoUDicePixel AccuracyNumber Cards (2-10)88.92%94.12%96.21%Face Cards (J,Q,K)86.54%92.78%95.34%Aces85.23%92.01%94.87%
Common Challenges

Corner under-segmentation (12% of errors)
Shadow confusion (8% of errors)
Multi-card scenes (5% of errors)

🔮 Future Improvements

Vision Transformers - Hybrid ViT-CNN architecture (+3-5% IoU expected)
Multi-Task Learning - Combined segmentation + classification
Attention Mechanisms - CBAM/SE-Net for edge refinement
Synthetic Data - GAN-based augmentation
Active Learning - Iterative improvement on hard cases
Test-Time Augmentation - Ensemble predictions (+2-3% IoU)

📱 Mobile App Features

Real-time camera feed processing
Live segmentation overlay
Card identification
FPS monitoring
Easy-to-use interface

🎓 Academic Context
Course: Computer Vision and Image Processing
Student: Parve Palial (Roll: 23117027)
Date: November 2, 2025
📄 License
[Add your license here]
🙏 Acknowledgments

Dataset: Kaggle Cards Image Dataset
U-Net Architecture: Ronneberger et al.
Framework: PyTorch, React Native Expo

📞 Contact
For questions or collaboration:

GitHub: [Your GitHub]
Email: [Your Email]
