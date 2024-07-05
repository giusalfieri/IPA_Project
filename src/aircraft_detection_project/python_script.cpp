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

    // The indentation is crucial in the following Python script
    const auto pythonScript = R"(
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from pathlib import Path
import sys

sco_file = Path(r'D:\IPA\Sources\IPA_Project-main\src\false_positive.txt')
csv_file = Path(r'D:\IPA\Sources\IPA_Project-main\src\roi_label_pairs.csv')
output_file = Path(r'D:\IPA\Sources\IPA_Project-main\src\output.csv')

# Read data from .sco file
with open(sco_file, 'r') as f:
    lines = f.readlines()

# Remove header and take the second column
sco_data = [line.split()[1] for line in lines[1:]]

# Read data from .csv file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    csv_data = [row[4] for row in reader]

# Write data to new CSV file in two columns
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Use zip to pair sco_data and csv_data
    for csv_val, sco in zip(csv_data, sco_data):
        writer.writerow([csv_val, sco])



# Generate x values
x = np.linspace(0, 2 * np.pi, 100)

# Generate y values (sine of x)
y = np.sin(x)

# Create a plot
plt.plot(x, y, label='Sine Wave')

# Add title and labels
plt.title('Simple Sine Wave Plot')
plt.xlabel('X values')
plt.ylabel('Sine of X')

# Add a legend
plt.legend()

# Display the plot
plt.show()
)";

    // Execute the Python script
    PyRun_SimpleString(pythonScript);

    // Close the Python interpreter
    Py_Finalize();
}
