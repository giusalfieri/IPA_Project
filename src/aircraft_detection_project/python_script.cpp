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
