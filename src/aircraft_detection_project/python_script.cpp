#include "python_script.h"
#include <filesystem>
#include <iostream>


/**
 * @brief Configures the Python environment and runs a Python script to generate a precision-recall curve.
 *
 * This function initializes the Python interpreter, configures the environment to include paths
 * from a specified virtual environment, and executes a Python script. The script reads positive
 * and negative scores from files, calculates the precision and recall, computes the AUC, and
 * plots the precision-recall curve.
 *
 * The steps are as follows:
 * 1. Check if the virtual environment exists.
 * 2. Initialize the Python interpreter.
 * 3. Add the virtual environment paths to `sys.path`.
 * 4. Execute the Python script.
 * 5. Finalize the Python interpreter.
 *
 * The Python script:
 * 1. Sets the source directory path from an environment variable.
 * 2. Reads positive and negative scores from `.sco` files.
 * 3. Loads the data into pandas DataFrames, adds labels, and combines them.
 * 4. Sorts the combined DataFrame by confidence scores.
 * 5. Calculates precision, recall, and AUC.
 * 6. Plots the precision-recall curve.
 *
 * @note This function assumes that the virtual environment path is defined by `VENV_PATH`
 *       and the source directory path by `SRC_DIR_PATH`.
 *
 * @see Py_Initialize
 * @see Py_Finalize
 * @see PySys_GetObject
 * @see PyList_Append
 * @see PyRun_SimpleString
 */
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
from sklearn.metrics import precision_recall_curve, auc

# Ottieni il percorso dalla variabile d'ambiente
os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')


# Leggi i percorsi dei file di input
positive_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_outputs', 'positive.sco').replace('\\', '/')
negative_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_outputs', 'negative.sco').replace('\\', '/')


# Funzione per caricare i dati dai file .sco saltando le righe di commento
def load_sco_file(filepath):
    return pd.read_csv(filepath, comment='#', sep=r'\s+', names=['sample', 'score'])

# Caricamento dei dati dai file .sco
positive_df = load_sco_file(positive_sco_path)
negative_df = load_sco_file(negative_sco_path)

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

# Calcolo dell'AUC con il metodo dei trapezi
auc_score = auc(recall, precision)


# Tracciamento della curva precision-recall
plt.plot(recall, precision, marker='.', label=f'AUC = {auc_score:.2f}')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='mediumpurple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
)";


    // Convert std::string to const char* and execute the Python script
    PyRun_SimpleString(pythonScript.c_str());

    // Close the Python interpreter
    Py_Finalize();
}