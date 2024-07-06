#include "python_script.h"
#include <filesystem>
#include <iostream>



void configureAndRunPythonScript()
{
    // Path to the virtual environment
    const std::string venvPath = VENV_PATH;

    // Check if the virtual environment exists
    if (!std::filesystem::exists(venvPath))
    {
        std::cerr << "Virtual environment not found at " << venvPath << "\n";
        return;
    }

    // Initialize Python interpreter
    Py_Initialize();

    // Imposta PYTHONPATH per includere i pacchetti dell'ambiente virtuale e altri percorsi necessari
    std::string pythonLibPath = venvPath + "/Lib";
    std::string pythonDLLPath = venvPath + "/DLLs";
    std::string pythonSitePackagesPath = venvPath + "/lib/site-packages";

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
import matplotlib.pyplot as plt
from sklearn.metrics import auc


os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')
print(f'SRC_DIR_PATH: {SRC_DIR_PATH}')


# Carica il file CSV
file_path = '/mnt/data/tuo_file.csv'
data = pd.read_csv(file_path, header=None, names=['label', 'score'])

# Supponiamo che il numero totale di ground truth objects sia noto
total_ground_truth_count = 100  # Sostituisci questo numero con il conteggio reale

# Ordina per confidence score in ordine decrescente
data = data.sort_values(by='score', ascending=False).reset_index(drop=True)

# Inizializza i conteggi cumulativi
TP_cumulative = 0
FP_cumulative = 0
precision = []
recall = []

# Itera attraverso le rilevazioni ordinate
for index, row in data.iterrows():
    if row['label'] == 'TP':
        TP_cumulative += 1
    elif row['label'] == 'FP':
        FP_cumulative += 1
    
    P = TP_cumulative / (TP_cumulative + FP_cumulative)
    R = TP_cumulative / total_ground_truth_count
    
    precision.append(P)
    recall.append(R)

# Crea un DataFrame con i risultati
pr_curve = pd.DataFrame({'precision': precision, 'recall': recall})

# Calcola l'AUC della precision-recall
pr_auc = auc(pr_curve['recall'], pr_curve['precision'])
print(f'Area Under Curve (AUC): {pr_auc}')

# Visualizza la curva precision-recall
plt.figure(figsize=(8, 6))
plt.plot(pr_curve['recall'], pr_curve['precision'], drawstyle='steps-post')
plt.fill_between(pr_curve['recall'], pr_curve['precision'], step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.show()
)";


    // Convert std::string to const char* and execute the Python script
    PyRun_SimpleString(pythonScript.c_str());

    // Close the Python interpreter
    Py_Finalize();
}