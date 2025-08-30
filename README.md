# Digit Recognition with LeNet

This project implements a **Convolutional Neural Network (CNN)** using the **LeNet architecture** to recognize handwritten digits from the MNIST dataset.  
It includes data preprocessing, training, evaluation, and experimentation with digit classification models.

---

## Features
- **LeNet-based CNN** (`lenet.py`, `LeNet1.py`)
- **MNIST data handling** and preprocessing scripts (`mnist.py`, `data.py`, `data_prep.py`)
- **Label creation utilities** (`make_labels.py`)
- **Evaluation pipeline** (`evaluation.py`) for model accuracy and loss tracking
- **Training & testing scripts** (`train.py`, `test.py`)
- Supports `DIGIT.npy` as a preprocessed dataset sample
- Modular, extensible structure for further deep learning experiments

---

## Project Structure
```plaintext
digit-recognition-lenet/
├── README.md               # Main project documentation
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore models, data dumps, etc.
├── src/                    # Source code
│   ├── lenet.py            # LeNet model implementation
│   ├── LeNet1.py           # Variant of LeNet
│   ├── data.py             # Data utilities
│   ├── data_prep.py        # Preprocessing pipeline
│   ├── digit_data.py       # Load and manage digit data
│   ├── mnist.py            # MNIST dataset loader
│   ├── make_labels.py      # Utility for creating labels
│   ├── evaluation.py       # Model evaluation metrics
│   ├── train.py            # Training script
│   └── test.py             # Testing script
├── data/                   # Small reference files
│   ├── README.md           # Dataset details & external links
│   ├── train_label.txt     # Training labels
│   ├── test_label.txt      # Testing labels
│   └── DIGIT.npy           # Sample digit dataset (preprocessed)
├── models/                 # Saved models (not committed)
└── outputs/                # Evaluation outputs (not committed)
