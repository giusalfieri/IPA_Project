# Top-Level CMakeLists.txt to build IPA project

cmake_minimum_required(VERSION 3.15)
project(SatelliteAircraftDetection)

# Hide unused predefined variables
set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE INTERNAL "")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/utils.cmake)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set up library and binary directories
if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT LIBRARY_OUTPUT_PATH)
if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
endif(NOT EXECUTABLE_OUTPUT_PATH)

# Set up OpenCV library
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV library not found or not properly installed")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Set up "Release" and "Debug" as default build mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING 
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." 
        FORCE) # Default to Release
endif()

# Disable annoying warnings on MSVC compilers
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    warnings_disable(CMAKE_CXX_FLAGS /wd4290) # vs2012: can't properly handle exceptions
    warnings_disable(CMAKE_CXX_FLAGS /wd4996) # vs2012: complains about unsafe standard C++ functions
    warnings_disable(CMAKE_CXX_FLAGS /wd4530) # vs2012: C++ exception handler used, but unwind semantics are not enabled
    warnings_disable(CMAKE_CXX_FLAGS /wd4503) # vs2012: decorated name length exceeded
endif()

# Disable annoying warning on Clang
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")
endif()

# Define folder where example images are stored
add_definitions(-DTRAINING_DATASET_PATH="${PROJECT_SOURCE_DIR}/dataset_training")
add_definitions(-DTESTING_DATASET_PATH="${PROJECT_SOURCE_DIR}/dataset_testing")
add_definitions(-DSRC_DIR_PATH="${PROJECT_SOURCE_DIR}")


# ---------------PYTHON STUFF --------------

# Trova l'interprete Python e i file di sviluppo
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include directory degli header di Python (dal sistema)
include_directories(${Python3_INCLUDE_DIRS})

# Link Python libraries directory (dal sistema)
link_directories(${Python3_LIBRARY_DIRS})

# Crea un ambiente virtuale nella directory specificata
set(VENV_PATH "${CMAKE_BINARY_DIR}/venv")

if(WIN32)
  set(PYTHON_VENV_EXECUTABLE "${VENV_PATH}/Scripts/python")
else()
  set(PYTHON_VENV_EXECUTABLE "${VENV_PATH}/bin/python")
endif()

# Aggiunge un target personalizzato per creare l'ambiente virtuale
add_custom_target(
    CreateVenv ALL
    COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_PATH}
    COMMENT "Creating virtual environment at ${VENV_PATH}"
)

# Aggiunge un target personalizzato per configurare l'ambiente virtuale e aggiornare pip e setuptools
add_custom_target(
    ConfigureVenv ALL
    COMMAND ${PYTHON_VENV_EXECUTABLE} -m pip install --upgrade pip setuptools wheel
    COMMENT "Configuring virtual environment at ${VENV_PATH}"
    DEPENDS CreateVenv
)

# Aggiunge un target personalizzato per installare i pacchetti richiesti
add_custom_target(
    InstallPackages ALL
    COMMAND ${PYTHON_VENV_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    COMMENT "Installing required Python packages"
    DEPENDS ConfigureVenv
)


add_definitions(-DVENV_PATH="${VENV_PATH}")

add_subdirectory(aircraft_detection_project)

# Assicurati che InstallPackages sia eseguito prima di costruire gli eseguibili o le librerie
add_dependencies(aircraft_detection_project InstallPackages)

# --------------- END PYTHON STUFF --------------


