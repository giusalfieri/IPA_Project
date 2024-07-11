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
import matplotlib.pyplot as plt
from sklearn.metrics import auc

os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')
print(f'SRC_DIR_PATH: {SRC_DIR_PATH}')

# Leggi il nome del file di ground truth dalla prima riga del file nella cartella detection
img_id_file_path = os.path.join(SRC_DIR_PATH, 'detection', 'testing_img_id.txt')
print(f'File di input: {img_id_file_path}')

with open(img_id_file_path, 'r') as file:
    ground_truth_filename = file.readline().strip() + '.txt'
print(f'Ground truth filename: {ground_truth_filename}')

# Carica il file .txt e conta le righe
ground_truth_file_path = os.path.join(SRC_DIR_PATH, 'dataset_testing', ground_truth_filename)

with open(ground_truth_file_path, 'r') as file:
    total_ground_truth_count = sum(1 for _ in file)
print(f'Total ground truth count: {total_ground_truth_count}')

# Carica il file CSV roi_label_pairs
csv_file_path = os.path.join(SRC_DIR_PATH, 'detection', 'roi_label_pairs.csv')
roi_label_pairs = pd.read_csv(csv_file_path, header=None, names=['x1', 'y1', 'x2', 'y2', 'label'])

# Carica il file true_positive.sco
sco_file_path = os.path.join(SRC_DIR_PATH, 'detection', 'true_positive.sco')
true_positive_sco = pd.read_csv(sco_file_path, sep=r'\s+', header=None, skiprows=1, names=['index', 'score'])
output_csv_path = os.path.join(SRC_DIR_PATH, 'detection', 'true_positive_sco.csv')

# Salva il DataFrame in un file CSV
true_positive_sco.to_csv(output_csv_path, index=False, header=False)

# Verifica che i due file abbiano lo stesso numero di righe
print(f'Numero di righe in roi_label_pairs: {len(roi_label_pairs)}')
print(f'Numero di righe in true_positive_sco: {len(true_positive_sco)}')
assert len(roi_label_pairs) == len(true_positive_sco), "I due file non hanno lo stesso numero di righe"

# Resetta gli indici di entrambi i DataFrame
roi_label_pairs.reset_index(drop=True, inplace=True)
true_positive_sco.reset_index(drop=True, inplace=True)

# Associa i due file
output_data = pd.DataFrame({
    'label': roi_label_pairs['label'],
    'score': true_positive_sco['score']
})

# Salva il risultato in un nuovo file
# Ordina output_data per score in ordine decrescente
output_data.sort_values(by='score', ascending=False, inplace=True)

# Salva il risultato in un nuovo file
output_file_path = os.path.join(SRC_DIR_PATH, 'detection', 'associated_labels_scores.csv')
output_data.to_csv(output_file_path, header=False, index=False)

# Inizializza i conteggi cumulativi
TP_cumulative = 0
FP_cumulative = 0
precision = []
recall = []

# Itera attraverso le rilevazioni ordinate
for index, row in output_data.iterrows():
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