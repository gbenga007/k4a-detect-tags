cmake_minimum_required(VERSION 3.24)
project(k4a_detect_tags LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(k4a REQUIRED)
find_package(k4arecord REQUIRED)
find_package(Open3D REQUIRED)
find_package(apriltag REQUIRED)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES
    k4a_detect_tags.cpp
)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(k4a_detect_tags  ${SOURCE_FILES})
target_link_libraries(k4a_detect_tags PUBLIC 
      
    ${Open3D_LIBRARIES} 
    k4a::k4arecord k4a::k4a
    apriltag::apriltag
    opencv_core opencv_imgproc opencv_highgui
    ${OpenCV_LIBS}
)
target_include_directories(k4a_detect_tags PUBLIC include)
