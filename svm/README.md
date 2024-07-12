# ucasML Installation Guide

> [!IMPORTANT]
> The `ucasML` program is designed to be run on a Linux system. Ensure you are using a compatible Linux distribution.

This guide will help you install and run the `ucasML` program on **Ubuntu** (and other distros as well, has been tested on Kali too). 

Follow the steps below to ensure all dependencies are met and the program runs successfully.

[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white&style=for-the-badge)](https://ubuntu.com/)
![Kali Linux 2024.1][(https://img.shields.io/badge/Kali_Linux-2024.1-0057A4?logo=kalilinux&logoColor=white&style=for-the-badge)](https://www.kali.org)







## Installation Steps

1. **Clone the Repository**

   First, clone the repository from GitHub:

   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```
   
2. **Make the ucasML Executable**

   Navigate to the bin directory inside the SVM folder and make the ucasML file executable (here ucasML is the name of the folder of the extracted ucasML package, change if needed):

   ```sh
   cd 'ucasML package'/bin
   chmod +x ucasML
   ```

3. **Set the Library Path Permanently**

   To make the library path setting permanent, you need to add the library path to your .bashrc (or .zshrc for zsh shell, or the config file of the shell you are using) file; here is done for bash shell:

   ```sh
   nano ~/.bashrc
   ```

   Add the following line at the end of the file (lib is the folder for extracted openCV ucasML, change accordingly if needed):

   ```sh
   export LD_LIBRARY_PATH=/home/your-username/Desktop/'ucasML package'/lib:$LD_LIBRARY_PATH
   ```
   
   Save the file and exit the text editor. Then, apply the changes (.bashrc for bash config file, change accordingly if using a different shell):

   
   ```sh
   source ~/.bashrc
   ```

4. **Check and Install Missing Dependencies**

   Use ldd to check for any missing dependencies:

   ```sh
   ldd ./ucasML
   ```
   Install any missing libraries as indicated by the ldd output. For example, if a library is missing, you can typically install it by installing the corresponding package using apt (or the package manager of the distro being used):

   ```sh
   sudo apt install <missing-library>
   ```

## Running the Program

Once all dependencies are resolved, you can run the ucasML program from the bin directory:

   ```sh
   ./ucasML
   ```





