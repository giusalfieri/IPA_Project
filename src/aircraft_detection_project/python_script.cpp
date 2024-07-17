#include "python_script.h"
#include <filesystem>
#include <iostream>



void configureAndRunPythonScript()
{
    // Path to the virtual environment
    const std::string venv_path = VENV_PATH;

    // Check if the virtual environment exists
    if (!std::filesystem::exists(venv_path))
    {
        std::cerr << "Virtual environment not found at " << venv_path << "\n";
        return;
    }

    // Initialize Python interpreter
    Py_Initialize();

    // Add the virtual environment paths to sys.path
    // This is necessary to import the required modules
    std::string pythonLibPath = venv_path + "/Lib";
    std::string pythonDLLPath = venv_path + "/DLLs";
    std::string pythonSitePackagesPath = venv_path + "/lib/site-packages";

    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString(pythonLibPath.c_str()));
    PyList_Append(sysPath, PyUnicode_FromString(pythonDLLPath.c_str()));
    PyList_Append(sysPath, PyUnicode_FromString(pythonSitePackagesPath.c_str()));

    // Path to the source directory
    const std::string srcDirPath = SRC_DIR_PATH;


    // The indentation is crucial in the following Python script
    const std::string pythonScript = R"(
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')
print(f'SRC_DIR_PATH: {SRC_DIR_PATH}')

# Leggi il nome del file di ground truth dalla prima riga del file nella cartella detection
positive_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_output', 'positive.sco')
negative_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_output', 'negative.sco')
print(f'File di input: {positive_sco_path}')
print(f'File di input: {negative_sco_path}')


# Caricamento dei dati dai file .sco
positive_df = pd.read_csv(positive_sco_path, header=None, names=['sample_id', 'score'])
negative_df = pd.read_csv(negative_sco_path, header=None, names=['sample_id', 'score'])

# Aggiunta delle etichette
positive_df['label'] = 1
negative_df['label'] = 0

# Unione dei due DataFrame
combined_df = pd.concat([positive_df, negative_df])

# Ordinamento per punteggio di confidenza in ordine decrescente
combined_df = combined_df.sort_values(by='score', ascending=False)

# Estrazione dei punteggi e delle etichette
scores = combined_df['score']
labels = combined_df['label']

# Calcolo di precision e recall
precision, recall, thresholds = precision_recall_curve(labels, scores)

# Tracciamento della curva precision-recall
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
)";


    // Convert std::string to const char* and execute the Python script
    PyRun_SimpleString(pythonScript.c_str());

    // Close the Python interpreter
    Py_Finalize();
}
