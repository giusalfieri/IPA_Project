<div align="center">
  <img src="doc/banner.png" style="width: 70%; height: auto;">
</div>

## Overview
This project focuses on detecting aircraft in satellite images using various machine learning and computer vision techniques. The workflow includes data preprocessing, feature extraction, clustering, and classification using Support Vector Machines (SVMs).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Steps](#workflow-steps)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The aim of this project is to develop an automated system for detecting aircraft in satellite imagery. The project utilizes a combination of image processing techniques, feature extraction methods, and machine learning algorithms to achieve high accuracy in aircraft detection.

## Features

- <span style="color: blue;">**Training Phase:**</span> Extract data from CSV for SVM training.
- <span style="color: green;">**Template Extraction:**</span> Extract templates from the dataset.
- <span style="color: red;">**Clustering:**</span> Perform K-Means clustering based on size and intensity.
- <span style="color: orange;">**Image Resizing:**</span> Resize images within each cluster to uniform dimensions.
- <span style="color: purple;">**Eigenplanes Generation:**</span> Generate eigenplanes for the clustered images.
- <span style="color: teal;">**Classification:**</span> Classify images using SVM and HOG features.
- <span style="color: brown;">**Performance Evaluation:**</span> Evaluate the performance of the classifier.


## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/aircraft-detection-satellite-images.git
    cd aircraft-detection-satellite-images
    ```

2. **Install dependencies:**
    Ensure you have OpenCV, CMake, and other necessary libraries installed. You can install the required Python packages using:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Build the project:**
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

2. **Run the project:**
    ```sh
    ./aircraft_detection [step1] [step2] ...
    ```
    For example, to run the training phase and extract templates:
    ```sh
    ./aircraft_detection training_phase extractTemplates
    ```

## Workflow Steps

### 1. Training Phase
Initiate the training phase for an SVM model by extracting data from a CSV file.
```sh
./aircraft_detection training_phase