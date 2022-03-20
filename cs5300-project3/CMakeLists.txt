cmake_minimum_required(VERSION 2.8)
project(cs5300-project3)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cs5300-project3 main.cpp)
target_link_libraries(cs5300-project3 ${OpenCV_LIBS})
