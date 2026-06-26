# 🐶 Dog Breed Image Classifier

This project uses pre-trained Convolutional Neural Networks (ResNet, AlexNet, and VGG) to classify pet images and identify dog breeds.

## 🚀 Features

- Image classification using PyTorch models
- Detection of dogs vs non-dogs
- Breed classification
- Comparison of multiple CNN architectures

## 🧠 Models Used

- ResNet
- AlexNet
- VGG

## 📂 Project Structure

- `check_images.py` → main script
- `classify_images.py` → image classification logic
- `get_pet_labels.py` → label extraction
- `adjust_results4_isadog.py` → dog identification
- `calculates_results_stats.py` → statistics calculation
- `print_results.py` → results output

## ▶️ How to Run

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

