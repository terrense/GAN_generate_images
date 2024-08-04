# GAN for Vehicle Image Generation

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate images of vehicles (e.g., vans, cars, SUVs) from aerial views.

![main structure](https://github.com/terrense/GAN_generate_images/blob/home-directory/GAN.jpg)

## Project Structure
```
my_gan_project/
├── data/
│   └── fake_images/
├── main.py
├── model.py
├── README.md
└── utils.py
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/my_gan_project.git
    cd my_gan_project
    ```

2. Install the required packages:
    ```bash
    pip install torch torchvision numpy
    ```

3. Prepare your dataset:
    - Place your vehicle images in `data/vehicles/`.

## Usage

Run the training script:
```bash
python main.py
