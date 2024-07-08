# ucasML Installation Guide

> [!IMPORTANT]
> The `ucasML` program is designed to be run on a Linux system. Ensure you are using a compatible Linux distribution.

This guide will help you install and run the `ucasML` program on **Ubuntu**. 
Follow the steps below to ensure all dependencies are met and the program runs successfully.






## Installation Steps

1. **Clone the Repository**

   First, clone the repository from GitHub:

   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```
   
2. **Make the ucasML Executable**

   Navigate to the bin directory inside the SVM folder and make the ucasML file executable:

   ```sh
   cd SVM/bin
   chmod +x ucasML
   ```

3. **Set the Library Path Permanently**

   To make the library path setting permanent, you need to add the library path to your .bashrc file.

   ```sh
   nano ~/.bashrc
   ```

   Add the following line at the end of the file:

   ```sh
   export LD_LIBRARY_PATH=/home/your-username/Desktop/SVM/opencv_libs_ucasML:$LD_LIBRARY_PATH
   ```
   
   Save the file and exit the text editor. Then, apply the changes:

   
   ```sh
   source ~/.bashrc
   ```

4. **Check and Install Missing Dependencies**

   Use ldd to check for any missing dependencies:

   ```sh
   ldd ./ucasML
   ```
   Install any missing libraries as indicated by the ldd output. For example, if a library is missing, you can typically install it using apt:

   ```sh
   sudo apt install <missing-library>
   ```

## Running the Program

Once all dependencies are resolved, you can run the ucasML program from the bin directory:

   ```sh
   ./ucasML
   ```





