# Real-Time Hand Gesture Recognition using YOLOv8

![Project Demo](https://i.imgur.com/your-demo-gif.gif) <!-- It's highly recommended to create a GIF of your output video and upload it here -->

This project implements a real-time hand gesture recognition system capable of identifying five distinct hand gestures (representing numbers 1 through 5). The system is built using the **YOLOv8 Nano** model, which is fine-tuned on a custom dataset for high performance and efficiency.

This work was developed as a project for a Deep Learning course.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training the Model](#2-training-the-model)
  - [3. Running Inference](#3-running-inference)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## About The Project

The goal of this project is to build an accurate and fast system for detecting hand gestures from a video feed. The core of the system is a YOLOv8n (Nano) object detection model, which is optimized for real-time performance.

The model was trained on a custom-built dataset comprising 543 images across 5 classes, annotated using Label Studio. The resulting system can process video files or a live webcam feed to identify and label hand gestures.

## Features

- **Real-Time Detection**: Utilizes the lightweight YOLOv8n model for high-speed inference.
- **Custom Trained**: Fine-tuned on a specialized dataset for recognizing 5 specific hand gestures.
- **High Accuracy**: Achieved good performance after approximately 10 hours of training.
- **Modular Code**: Includes scripts for video processing, model training, and inference.

---

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

This project uses Python 3.10+. You will also need `pip` to install the required packages.

- **Python & pip**:
  ```bash
  sudo apt-get install python3.10-venv
  ```
- **PyTorch**: Installation instructions vary by system. Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct command for your setup (e.g., with or without CUDA).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/armankhl/yolov8-hand-gesture-recognition.git
    cd your-repo-name
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the [Roadmap](#-roadmap-for-a-professional-repository) section for details.)*

---

## Usage

The project is divided into three main stages: data preparation, training, and inference.

### 1. Data Preparation

The `video_capture.ipynb` notebook is used to extract frames from video files.

-   Update the `video_path` to your source video.
-   Set the `output_dir` where the frames will be saved.
-   Run the cells to generate the image dataset.
-   After extraction, use a labeling tool like [Label Studio](https://labelstud.io/) to annotate the images in YOLO format.

### 2. Training the Model

The `yolo_train_and_testing.ipynb` notebook is designed for training in a Google Colab environment.

-   **Upload Dataset**: Upload your zipped dataset (containing `train/` and `val/` folders with `images/` and `labels/` subdirectories) to Google Drive.
-   **Configure Paths**: Update the paths in the notebook to point to your dataset and desired checkpoint directory.
-   **Run Training**: Execute the cells sequentially. The script will:
    1.  Mount Google Drive.
    2.  Unzip the dataset.
    3.  Create a `data.yaml` configuration file.
    4.  Load the pre-trained YOLOv8n model.
    5.  Start the fine-tuning process, saving checkpoints periodically.

### 3. Running Inference

The final cells in `yolo_train_and_testing.ipynb` show how to run detection on a video.

-   Load your trained model weights (`.pt` file).
-   Set the `video_path` to the input video you want to process.
-   Set the `output_video_path` for the result.
-   Run the script to generate a new video with bounding boxes and labels overlaid on the detected gestures.

---

## Model Details

-   **Architecture**: YOLOv8 Nano (`yolov8n.pt`)
-   **Total Classes**: 5 ('One', 'Two', 'Three', 'Four', 'Five')
-   **Training Time**: Approx. 10 hours
-   **Dataset Size**: 543 images

The model achieves good accuracy in identifying the target gestures, making it suitable for real-time applications.

## Dataset

The dataset was created from two sources:
1.  A publicly available dataset from Mendeley Data: [Hand Gesture Recognition Database](https://data.mendeley.com/datasets/ndrczc35bt/1).
2.  Custom-recorded videos to ensure diversity and robustness.

All images were manually annotated using Label Studio.

---

## Acknowledgments

-   Credit to the authors of the [Mendeley dataset](https://data.mendeley.com/datasets/ndrczc35bt/1) for providing the initial image base.
-   The [Ultralytics](https://github.com/ultralytics/ultralytics) team for the YOLOv8 model.