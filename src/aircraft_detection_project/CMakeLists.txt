# CMakeLists.txt per aircraft_detection_project

# Include libraries
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/ipalib)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/ucaslib)
include_directories(${Python3_INCLUDE_DIRS})

# Find sources
file(GLOB src *.h *.hpp *.cpp)

# Fix va_start error in VS 14+
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# Create executable from sources
add_executable(aircraft_detection_project ${src})

# Link the executable to other modules / libraries
target_link_libraries(aircraft_detection_project ipalib ucaslib ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Python3_LIBRARIES})

# Include directories
target_include_directories(aircraft_detection_project PUBLIC ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
