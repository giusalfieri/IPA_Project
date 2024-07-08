<div align="center">
  <img src="doc/banner.png" style="width: 50%; height: auto;">
</div>



## 🌍 Overview 
This project focuses on detecting aircraft in satellite images using various machine learning and computer vision techniques. The workflow includes data preprocessing, feature extraction, clustering, and classification using Support Vector Machines (SVMs).

---

## 📚 Table of Contents 

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Steps](#workflow-steps)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

The aim of this project is to develop an automated system for detecting aircraft in satellite imagery. The project utilizes a combination of image processing techniques, feature extraction methods, and machine learning algorithms to achieve high accuracy in aircraft detection.

---

## ⚙️ Features

- **Training Phase:** Extract data from CSV for SVM training.
- **Template Extraction:** Extract templates from the dataset.
- **Clustering:** Perform K-Means clustering based on size and intensity.
- **Image Resizing:** Resize images within each cluster to uniform dimensions.
- **Eigenplanes Generation:** Generate eigenplanes for the clustered images.
- **Classification:** Classify images using SVM and HOG features.
- **Performance Evaluation:** Evaluate the performance of the classifier.

> [!NOTE]  
> For a detailed description, see [project_report.pdf](project_report.pdf).

---

## 📦 Dependencies
(Section to be completed)
Ensure you have the following dependencies installed on your system:

- [**OpenCV**](https://opencv.org/releases/)
- [**CMake**](https://cmake.org/): For building the project. 
- **Python 3.x**
  

---
## 🛠️ Installation 

1. **Clone the repository:**
    ```sh
    git clone https://github.com/giusalfieri/IPA_Project.git
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
    ./aircraft_detection_project [step1] [step2] ...
    ```
    For example, to run the training phase and extract templates:
    ```sh
    ./aircraft_detection_project training_phase extractTemplates
    ```

---

## ➡️ ✅ ➡️ ✅ Workflow Steps 

### 1. Training Phase
Initiate the training phase for an SVM model by extracting data from a CSV file.
```sh
./aircraft_detection training_phase
```
---

##  👥 Contributors

| Nome              |  GitHub                                               |
|-------------------|-------------------------------------------------------|
| Giuseppe Alfieri  | [@giusalfieri](https://github.com/giusalfieri)        |
| Paolo Simeone     | [@bonoboprog](https://github.com/bonoboprog)          |
| Aurora Pisa       | [@aurorapisa](https://github.com/aurorapisa)          |
| Riccardo D'Aguanno| [@ricdag8](https://github.com/ricdag8)                |
| Gianmarco Luongo  | [@GianmarcoL](https://github.com/GianmarcoL)          |

---


##  📜 License

This project is licensed under the MIT License. See the [LICENSE](License.txt) file for details.

