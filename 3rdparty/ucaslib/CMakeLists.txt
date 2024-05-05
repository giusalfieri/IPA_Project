# disable annoying warnings on MSVC compilers
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	warnings_disable(CMAKE_CXX_FLAGS /wd4290) # vs2012: can't properly handle exceptions
	warnings_disable(CMAKE_CXX_FLAGS /wd4996) # vs2012: complains about unsafe standard C++ functions
	warnings_disable(CMAKE_CXX_FLAGS /wd4530) # vs2012: C++ exception handler used, but unwind semantics are not enabled
	warnings_disable(CMAKE_CXX_FLAGS /wd4503) # vs2012: decorated name length exceeded
endif()

file(GLOB common_src *.h *.cpp *.hpp)

add_definitions(-D_CRT_NO_VA_START_VALIDATION)

option (WITH_GDCM "Use GDCM to read DICOM images" OFF)
if(WITH_GDCM)
	message(STATUS "Include GDCM ...")
	find_package(GDCM REQUIRED)
	if(NOT GDCM_FOUND)
	    message(FATAL_ERROR "GDCM library not found or not properly installed")
	endif()
	message(STATUS "Include GDCM ... OK!")
	add_definitions(-DWITH_GDCM)
	INCLUDE(${GDCM_USE_FILE})
	add_library(ucaslib STATIC ${common_src})
	target_link_libraries(ucaslib gdcmMSFF)
else()
	add_library(ucaslib STATIC ${common_src})
endif(WITH_GDCM)