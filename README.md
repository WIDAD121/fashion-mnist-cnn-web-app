# Fashion-MNIST CNN Classifier — Flask Web App

A web application built with **Flask** and **TensorFlow/Keras** that lets users train a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset, inspect training results, and classify clothing images.

The application interface is in French.

## Features

- Upload training and test datasets in CSV format.
- Preview dataset rows and dimensions.
- Configure CNN training parameters, including convolution filters, kernel size, pooling size, optimizer, loss function, batch size, and number of epochs.
- Train a CNN to classify Fashion-MNIST clothing images.
- Display evaluation metrics: accuracy, precision, recall, and F1-score.
- Visualize the confusion matrix, class-level accuracy, training accuracy, and loss.
- Upload an image for prediction after a model has been trained.

## Fashion-MNIST classes

The classifier uses the 10 Fashion-MNIST categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Technologies

- Python
- Flask
- TensorFlow / Keras
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn
- Pillow (PIL)
- HTML and CSS

## Repository structure

```text
fashion-mnist-cnn-web-app/
├── README.md
├── requirements.txt
├── .gitignore
└── app/
    ├── main.py
    ├── static/
    ├── templates/
    └── uploads/
        └── .gitkeep
```

## Setup and run

Python 3.10 or 3.11 is recommended. TensorFlow compatibility depends on your Python version and operating system.

1. Clone the repository:

   ```bash
   git clone https://github.com/WIDAD121/fashion-mnist-cnn-web-app.git
   cd fashion-mnist-cnn-web-app
   ```

2. Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application:

   ```bash
   cd app
   python main.py
   ```

5. Open the local address printed by Flask (usually `http://127.0.0.1:5000/`).

## Dataset format

The application expects separate training and test CSV files to be uploaded through its interface.

- The **first column** contains the class label (0–9).
- The remaining **784 columns** contain the pixel values for a 28 × 28 grayscale image.

The original archive included large CSV files. They are intentionally not committed here because the training CSV is larger than GitHub's 100 MB per-file limit. Add the dataset locally in the application's `app/uploads/` folder or upload it through the application. Do not commit large datasets or private user uploads to GitHub.

## Important notes

- The prediction page requires a model to have been trained in the current application session first.
- This repository contains the application source and interface assets, not a pre-trained model.
- The original source runs Flask with `debug=True` and uses a hard-coded secret key. These settings are for local academic development only. Before deploying publicly, disable debug mode and load a strong secret key from an environment variable.
- Review and test dependency compatibility in your own environment before running the application.

## Academic project

This repository contains an academic mini-project focused on applying Convolutional Neural Networks to image classification with the Fashion-MNIST dataset, through a Flask web interface.
