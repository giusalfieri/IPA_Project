<p align="center">
  <img src="docs/banner.png" style="width: 50%; height: auto;">
</p>

<p align="center" style="margin-top: 40px;">
    <img src="https://img.shields.io/github/stars/giusalfieri/IPA_Project?style=plastic&color=00bfff&labelColor=655ff0" alt="GitHub stars">
    <img src="https://img.shields.io/github/contributors/giusalfieri/IPA_Project?style=plastic&color=00bfff&labelColor=655ff0" alt="GitHub contributors">
    <img src="https://img.shields.io/github/repo-size/giusalfieri/IPA_Project?style=plastic&color=00bfff&labelColor=655ff0" alt="GitHub repo size">
    <img src="https://img.shields.io/github/license/giusalfieri/IPA_Project?style=plastic&color=00bfff&labelColor=655ff0" alt="GitHub License">
</p>

## <a name="overview">🌍 Overview</a>

This project focuses on detecting aircraft in satellite images using various Computer Vision and Machine Learning techniques. The workflow includes data preprocessing, feature extraction, clustering, and classification using Support Vector Machines (SVMs).

> [!NOTE]  
> The SVM used in this project is part of the **ucasML** *command-line interface (CLI)* tool.
> <p align="center">
> <img src="ucasML_package/assets/ucasML_banner.png" alt="ucasML Logo" style="width: 25%; height: auto; margin-top: 20px;">
> </p>

> [!IMPORTANT]  
> For detailed instructions on installing **ucasML** tool on your machine, please refer to the *ad hoc* [README](ucasML_package/README.md).

---

## <a name="api-documentation">📖 API Documentation</a>

For a detailed reference of all functions and classes, please refer to the [API Documentation](https://giusalfieri.github.io/IPA_Project/).

---

## 📚 Table of Contents 

- [Introduction](#introduction)
- [Features](#features)
- 📦[Dependencies](#dependencies)
- 🔨[Installation](#installation)
- [Usage](#usage)
- ➡️ ✅ ➡️ ✅[Pipeline](#pipeline)
- 👥[Contributors](#contributors)
- 📜[License](#license)

---

## Introduction

The aim of this project is to develop an automated system for detecting aircraft in satellite imagery. The project utilizes a combination of image processing techniques, feature extraction methods, and machine learning algorithms to achieve high accuracy in aircraft detection.

---

## <a name="features">⚙️ Features</a>

- **Template Extraction:** Extract templates from the training dataset.
- **Clustering:** Perform K-Means clustering based on size and intensity.
- **Image Resizing:** Resize images within each cluster to uniform dimensions.
- **Eigenplanes Generation:** Generate eigenplanes for the clustered images.
- **SVM Cross Validation:** Extract data from CSV for SVM training..
- **Performance Evaluation:** Evaluate the performance of the classifier.

> [!NOTE]  
> The training dataset used is a sub-set of the training dataset of [HRPlanesv2 Data Set](https://github.com/dilsadunsal/HRPlanesv2-Data-Set).

> [!IMPORTANT]  
> For a detailed description of the underlying ***ratio*** of design choices taken, see [project_report.pdf](docs/project_proposal/main.pdf).

---

## <a name="dependencies">📦 Dependencies</a>

[![OpenCV](https://img.shields.io/badge/-OpenCV-black?style=for-the-badge&logoColor=white&logo=opencv&color=blue)](https://opencv.org/releases/)
[![CMake](https://img.shields.io/badge/-CMake-black?style=for-the-badge&logoColor=white&logo=cmake&color=4CAF50)](https://cmake.org/)
[![C++20](https://img.shields.io/badge/-C++20-black?style=for-the-badge&logo=c%2B%2B&logoColor=white&color=red)](https://isocpp.org/)
[![Python 3.x](https://img.shields.io/badge/-Python%203.x-black?style=for-the-badge&logoColor=white&logo=python&color=yellow)](https://www.python.org/)

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

<details>
<summary>***Command Line Instructions***</summary>

 1. **Clone the repository:**
    ```sh
    git clone https://github.com/giusalfieri/IPA_Project.git
    cd IPA_Project
    ```

2. **Create a build directory:**
    ```sh
    mkdir build
    cd build
    ```

3. **Configure the project using CMake:**
    ```sh
    cmake ..
    ```

4. **Compile the project:**
    ```sh
    make
    ```

5. **Run the project:**
    ```sh
    ./aircraft_detection_project [step1] [step2] ...
    ```
    For example, to run the entire pipeline:
    ```sh
    ./aircraft_detection_project training_phase extractTemplates KMeansBySize KMeansByIntensity resizeImagesInClusters generateEigenplanes extract_SVM_Training_Data Performance_evaluation
    ```
    Or you can run individual steps as needed. See the [Pipeline](#pipeline) section for detailed steps.
   
</details>

![](./docs/render1721142899171.gif)


<details>
<summary>***CMake GUI and Visual Studio Instructions***</summary>
  
1. **Clone the repository:**
    ```sh
    git clone https://github.com/giusalfieri/IPA_Project.git
    cd IPA_Project
    ```

2. **Open CMake GUI:**
    - Set the "Where is the source code" field to the path of the `IPA_Project` directory.
    - Set the "Where to build the binaries" field to a new `build` directory within the `IPA_Project` directory (e.g., `IPA_Project/build`).

3. **Configure the project:**
    - Click on the "Configure" button.
    - Select your version of Visual Studio (e.g., Visual Studio 2019) and the appropriate platform (e.g., x64).
    - Click "Finish" to start the configuration process.
    - If there are any missing dependencies, resolve them and click "Configure" again.

4. **Generate the project files:**
    - Once the configuration is complete, click the "Generate" button to create the Visual Studio solution and project files.

5. **Open the project in Visual Studio:**
    - Navigate to the `build` directory and open the generated `.sln` file (e.g., `IPA_Project.sln`) with Visual Studio.

6. **Build the project:**
    - In Visual Studio, set the build configuration to `Release` or `Debug` as needed.
    - Build the solution by selecting "Build Solution" from the "Build" menu.

7. **Run the project:**
    - In Visual Studio, set the startup project to `aircraft_detection_project`.
    - Configure the project properties if needed to include any command-line arguments.
    - Start debugging or run the project without debugging as needed.
 
</details>

For example, to run the entire pipeline, you can set the command-line arguments in the project properties as:

```sh
./aircraft_detection_project training_phase extractTemplates KMeansBySize KMeansByIntensity resizeImagesInClusters generateEigenplanes extract_SVM_Training_Data Performance_evaluation
```

Or you can run individual steps as needed. See the [Pipeline](#pipeline) section for detailed steps.

## <a name="pipeline">➡️ ✅ ➡️ ✅ Pipeline</a>
> [!NOTE]
> In the following lines, workflow steps instructions are given and explained.
> After each phase is concluded, meaning that each command given has terminated execution, a corresponding file with a .done extension is created in /src/steps_completed folder.
> While given a command, the program checks if the previous step has been done, checking the existance of the corresponding file in /src/steps_completed folder; if not, a warning message indicating the missing step is printed out.

### 1. Training Phase
Initiate the training phase for an SVM model by extracting data from a CSV file.
The extracted data is prepared and formatted to be used for training the SVM algorithm.
```sh
./aircraft_detection_project training_phase
```
---
### 2. Template extraction Phase
Extract templates from the dataset, which are necessary for further processing and analysis.
This can be done by executing the command:
```sh
./aircraft_detection_project extractTemplates
```
---
### 3. K-means-by-size clustering Phase
Apply the K-Means clustering algorithm to group extracted templates, treated as data points, based on their size; templates with similar dimensions are grouped into clusters.
Do this by executing the following command:
```sh
./aircraft_detection_project KMeansBySize
```
---
### 4. K-means-by-intensity clustering Phase
Apply the K-Means clustering algorithm to group extracted templates, treated as data points, based on their intensity; templates with similar intensity values are grouped into clusters.
This can be done by executing:
```sh
./aircraft_detection_project KMeansByIntensity
```
---
### 5. Images resizing Phase 
All the images in each cluster need to have the same dimensions as this is necessary for generating eigenplanes (see next phase).
In order to do this, execute as follows: 
```sh
./aircraft_detection_project resizeImagesInClusters
```
---
### 6. Eigenplanes generating Phase 
Generate Eigenplanes for each cluster of images previously resized.
By "Eigenplanes" is meant the outcome of the application of the EigenFace algorithm.
The details of the algorithm can be read here: https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html
Execute the following command:
```sh
./aircraft_detection_project generateEigenplanes
```
---
### 7. Detection Phase 
This is the most important step.
Classify the given testing image based on the features extracted in the previous phase (HOG features).
This phase consists in performing template matching, classifying points by YOLO boxes, extracting ROIs and finally using a SVM model to classify the ROIs.
In SVM training, 10 fold cross validation is done, and the outcome are two .sco files, relative to true positives and negatives, respectively; these two files must then be placed in /src/svm_cv_outputs directory.
Every row in these files is a sample (candidate region) with a classification score associated, columns being sample id and score. 
Detection is done by executing:
```shs
./aircraft_detection_project extract_SVM_Training_Data
```
---
### 8. Performance evaluation phase 
Evaluate the performance of detection done in the previous step by running a Python script, to obtain the Precision-Recall curve and the average precision, calculated with trapezoidal method.
```sh
./aircraft_detection_project Performance_evaluation
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

