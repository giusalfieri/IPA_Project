<div align="center">
  <img src="doc/banner.png" style="width: 50%; height: auto;">
</div>

<p align="center" style="margin-top: 20px;">
    <img src="https://img.shields.io/github/stars/octocat/Hello-World?style=plastic&color=ff69b4&labelColor=8a2be2" alt="GitHub stars">
    <img src="https://img.shields.io/github/contributors/octocat/Hello-World?style=plastic&color=ff69b4&labelColor=8a2be2" alt="GitHub contributors">
    <img src="https://img.shields.io/github/repo-size/octocat/Hello-World?style=plastic&color=ff69b4&labelColor=8a2be2" alt="GitHub repo size">
</p>

## <a name="overview">🌍 Overview</a>

This project focuses on detecting aircraft in satellite images using various machine learning and computer vision techniques. The workflow includes data preprocessing, feature extraction, clustering, and classification using Support Vector Machines (SVMs).

> [!IMPORTANT]  
> For detailed instructions on installing the SVM, please refer to the *ad hoc* [README](ucasML_package/README.md).

---

## 📚 Table of Contents 

- [Introduction](#introduction)
- [Features](#features)
- 📦[Dependencies](#dependencies)
- 🔨[Installation](#installation)
- [Usage](#usage)
- ➡️ ✅ ➡️ ✅[Workflow Steps](#workflow-steps)
- 👥[Contributors](#contributors)
- 📜[License](#license)

---

## Introduction

The aim of this project is to develop an automated system for detecting aircraft in satellite imagery. The project utilizes a combination of image processing techniques, feature extraction methods, and machine learning algorithms to achieve high accuracy in aircraft detection.

---

## <a name="features">⚙️ Features</a>

> [!NOTE]  
> For a detailed description of the underlying ***ratio*** of design choices taken, see [project_report.pdf](doc/project_proposal/main.pdf).

- **Training Phase:** Extract data from CSV for SVM training.
- **Template Extraction:** Extract templates from the dataset.
- **Clustering:** Perform K-Means clustering based on size and intensity.
- **Image Resizing:** Resize images within each cluster to uniform dimensions.
- **Eigenplanes Generation:** Generate eigenplanes for the clustered images.
- **Classification:** Classify images using SVM and HOG features.
- **Performance Evaluation:** Evaluate the performance of the classifier.

---

## <a name="dependencies">📦 Dependencies</a>

[![OpenCV](https://img.shields.io/badge/-OpenCV-black?style=for-the-badge&logoColor=white&logo=opencv&color=blue)](https://opencv.org/releases/)
[![CMake](https://img.shields.io/badge/-CMake-black?style=for-the-badge&logoColor=white&logo=cmake&color=4CAF50)](https://cmake.org/)
[![Python 3.x](https://img.shields.io/badge/-Python%203.x-black?style=for-the-badge&logoColor=white&logo=python&color=yellow)](https://www.python.org/)
[![C++20](https://img.shields.io/badge/-C++20-black?style=for-the-badge&logo=c%2B%2B&logoColor=white&color=red)](https://isocpp.org/)

Ensure you have the above installed on your machine:
  

> [!IMPORTANT]  
> Make sure Python is installed system wide (e.g., on Windows, Python must be added to the system `PATH`).

> [!TIP]  
> Click the badges to visit their official websites.
> 
> For a list of C++20 compliant compilers, click [here](https://en.cppreference.com/w/cpp/compiler_support).

---

## <a name="installation">🔨 Installation</a>

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
![](./doc/render1721142899171.gif)

2. **Run the project:**
    ```sh
    ./aircraft_detection_project [step1] [step2] ...
    ```
    For example, to run the training phase and extract templates:
    ```sh
    ./aircraft_detection_project training_phase extractTemplates
    ```

---

## <a name="workflow-steps">➡️ ✅ ➡️ ✅ Workflow Steps</a>

### 1. Training Phase
Initiate the training phase for an SVM model by extracting data from a CSV file.
```sh
./aircraft_detection_project training_phase
```
---

## <a name="contributors">👥 Contributors</a>

| Name              |  GitHub                                               |
|-------------------|-------------------------------------------------------|
| Giuseppe Alfieri  | [@giusalfieri](https://github.com/giusalfieri)        |
| Paolo Simeone     | [@bonoboprog](https://github.com/bonoboprog)          |
| Aurora Pisa       | [@aurorapisa](https://github.com/aurorapisa)          |
| Riccardo D'Aguanno| [@ricdag8](https://github.com/ricdag8)                |
| Gianmarco Luongo  | [@GianmarcoL](https://github.com/GianmarcoL)          |

---

## <a name="license">📜 License</a> 

This project is licensed under the MIT License. See the [LICENSE](License.txt) file for details.

