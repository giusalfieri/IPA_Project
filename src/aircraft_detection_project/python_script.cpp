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
from sklearn.metrics import precision_recall_curve, auc

# Ottieni il percorso dalla variabile d'ambiente
os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')
print(f'SRC_DIR_PATH: {SRC_DIR_PATH}')

# Leggi i percorsi dei file di input
positive_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_output', 'positive.sco').replace('\\', '/')
negative_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_output', 'negative.sco').replace('\\', '/')
print(f'File di input: {positive_sco_path}')
print(f'File di input: {negative_sco_path}')

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
print(f'AUC: {auc_score:.2f}')

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
