# Top-Level CmakeLists.txt to build IPA project

# Define a variable for the project name
set(PROJECT_NAME_VAR SatelliteAircraftDetection)
project(${PROJECT_NAME_VAR})
cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0054 OLD) 
cmake_policy(SET CMP0045 OLD)

# hide unused predefined variables
set (CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE INTERNAL "")

# include cmake scripts
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/utils.cmake)

# require C++11
set (CMAKE_CXX_STANDARD 11)

# set up library and binary directories
if(NOT LIBRARY_OUTPUT_PATH)
  set (LIBRARY_OUTPUT_PATH ${${PROJECT_NAME_VAR}_BINARY_DIR}/libs CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT LIBRARY_OUTPUT_PATH)
if(NOT EXECUTABLE_OUTPUT_PATH)
  set (EXECUTABLE_OUTPUT_PATH ${${PROJECT_NAME_VAR}_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
endif(NOT EXECUTABLE_OUTPUT_PATH)

# set up OpenCV library
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV library not found or not properly installed")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# set up Boost library
#set(Boost_DEBUG ON)
#set(BOOST_PATH "" CACHE PATH "Boost library path")
#set(ENV{BOOST_ROOT} ${BOOST_PATH})
#find_package(Boost REQUIRED)
#include_directories( ${Boost_INCLUDE_DIR} )

# set up "Release" and "Debug" as default build mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING 
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." 
        FORCE) # Default to Release
endif()

# disable annoying warnings on MSVC compilers
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	warnings_disable(CMAKE_CXX_FLAGS /wd4290) # vs2012: can't properly handle exceptions
	warnings_disable(CMAKE_CXX_FLAGS /wd4996) # vs2012: complains about unsafe standard C++ functions
	warnings_disable(CMAKE_CXX_FLAGS /wd4530) # vs2012: C++ exception handler used, but unwind semantics are not enabled
	warnings_disable(CMAKE_CXX_FLAGS /wd4503) # vs2012: decorated name length exceeded
endif()

# disable annoying warning on Clang
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")
endif()


# define folder where example images are stored
add_definitions(-DDATASET_PATH="${${PROJECT_NAME_VAR}_SOURCE_DIR}/dataset")

# build modules
add_subdirectory( 3rdparty )
add_subdirectory( aircraft_detection_project )