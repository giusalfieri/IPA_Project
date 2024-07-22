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
import numpy as np
import matplotlib.pyplot as plt
import os

GROUNDTRUTH_SIGNS = 384

# Ottieni il percorso dalla variabile d'ambiente
os.environ['SRC_DIR_PATH'] = ')" + srcDirPath + R"('
SRC_DIR_PATH = os.getenv('SRC_DIR_PATH', '.')

# Costruisci i percorsi dei file
positive_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_outputs', 'positive.sco')
negative_sco_path = os.path.join(SRC_DIR_PATH, 'svm_cv_outputs', 'negative.sco')

# Verifica se i file esistono
if not os.path.isfile(positive_sco_path):
    print(f"File non trovato: {positive_sco_path}")
    exit(1)

if not os.path.isfile(negative_sco_path):
    print(f"File non trovato: {negative_sco_path}")
    exit(1)

print(f'SRC_DIR_PATH: {SRC_DIR_PATH}')
print(f'positive score file path: {positive_sco_path}')
print(f'negative score file path: {negative_sco_path}')

# Leggi i file
with open(positive_sco_path, 'r') as file_tp, open(negative_sco_path, 'r') as file_fp:
    tp_data = file_tp.readlines()
    fp_data = file_fp.readlines()

# Analizza i punteggi e ignora l'intestazione
tp_scores = np.array([float(line.split('\t')[1].strip()) for line in tp_data[1:]])
fp_scores = np.array([float(line.split('\t')[1].strip()) for line in fp_data[1:]])

# Crea etichette (1 per TPs e 0 per FPs)
tp_labels = np.ones_like(tp_scores)
fp_labels = np.zeros_like(fp_scores)

# Combina punteggi e etichette
scores = np.concatenate((tp_scores, fp_scores))
labels = np.concatenate((tp_labels, fp_labels))

# Ottieni gli indici che ordinerebbero i punteggi in ordine decrescente
sorted_indices = np.argsort(scores)[::-1]

# Usa questi indici per ordinare entrambi gli array
scores_sorted = scores[sorted_indices]
labels_sorted = labels[sorted_indices]

cumulativeTP = 0
cumulativeFP = 0
precision = []
recall = []

for score, label in zip(scores_sorted, labels_sorted):
    if label:
        cumulativeTP += 1
    else:
        cumulativeFP += 1

    precision.append(cumulativeTP / (cumulativeTP + cumulativeFP))
    recall.append(cumulativeTP / GROUNDTRUTH_SIGNS)

avg_precision = precision[0] * recall[0]
for i in range(1, len(recall)):
    avg_precision += (recall[i] - recall[i - 1]) * precision[i]

print("avg precision: " + str(avg_precision))

path = os.path.dirname(positive_sco_path) + '/'
plt.plot(recall, precision)
plt.savefig(os.path.join(path, f"{avg_precision:.2f}.png"))
plt.show()

)";


    // Convert std::string to const char* and execute the Python script
    PyRun_SimpleString(pythonScript.c_str());

    // Close the Python interpreter
    Py_Finalize();
}
