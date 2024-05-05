# include libraries
include_directories (${${PROJECT_NAME_VAR}_SOURCE_DIR}/3rdparty/ipalib)
include_directories (${${PROJECT_NAME_VAR}_SOURCE_DIR}/3rdparty/ucaslib)

# find sources
file(GLOB src *.h *.hpp *.cpp)

# fix va_start error in VS 14+
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# create executable from sources
add_executable(aircraft_detection_project ${src})

# link the executable to other modules / libraries 
target_link_libraries(aircraft_detection_project ipalib ucaslib ${OpenCV_LIBS} ${Boost_LIBRARIES})
