🧠 VGG-16 Transfer Learning using PyTorch
📌 Overview

This project demonstrates Transfer Learning using the VGG-16 architecture in PyTorch for image classification.
A pretrained VGG-16 model is used as a feature extractor, and custom classification layers are added to adapt the network to a new dataset.

The pipeline covers data loading, preprocessing, model customization, training, and evaluation using PyTorch best practices.

🧠 Training Pipeline Architecture

The workflow implemented in this project follows these steps:

Dataset Preparation

Image dataset organized into training and validation folders

Image preprocessing and augmentation

Data Loading

Custom transformations using torchvision.transforms

Efficient batching using DataLoader

Model Architecture

Pretrained VGG-16 model loaded from torchvision.models

Frozen convolutional layers

Custom fully connected classifier added

Training Loop

Forward pass

Loss computation

Backpropagation

Optimizer step

Validation after each epoch

Evaluation

Accuracy and loss monitoring

Model performance comparison across epochs

📊 Model & Training Details

- Problem Type: Image Classification

- Base Model: VGG-16 (Pretrained on ImageNet)

- Framework: PyTorch

- Loss Function: CrossEntropyLoss

- Optimizer: Adam / SGD

- Training Strategy: Transfer Learning

- Device Support: CPU / GPU

🛠 Technologies Used

- Python

- PyTorch

- Torchvision

- NumPy

- Matplotlib

- Google Colab / Jupyter Notebook

📂 Dataset

Type: Image classification dataset

Structure:

dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...


Loading Method: torchvision.datasets.ImageFolder

Preprocessing: Resizing, normalization, and augmentation

📁 Project Structure
VGG16-Transfer-Learning/
│
├── VGG_16_Transfer_Learning.ipynb   # Main training notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies

▶ How to Run
1️⃣ Clone the repository
git clone https://github.com/your-username/VGG16-Transfer-Learning.git
cd VGG16-Transfer-Learning

2️⃣ Install dependencies
pip install torch torchvision numpy matplotlib

3️⃣ Open the notebook
jupyter notebook VGG_16_Transfer_Learning.ipynb

4️⃣ Run all cells

Ensure dataset paths are correctly set

Enable GPU if running on Google Colab

🚀 Key Learning Outcomes

Understanding Transfer Learning

Using pretrained CNN models

Freezing and fine-tuning layers

Building efficient PyTorch training loops

Working with real-world image datasets

✨ Author

Yash Prajapati
M.Tech (Artificial Intelligence)
