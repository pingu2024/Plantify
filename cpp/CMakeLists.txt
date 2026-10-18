cmake_minimum_required(VERSION 3.16)
project(plantify_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_executable(plantify_infer infer.cpp)
target_link_libraries(plantify_infer PRIVATE ${OpenCV_LIBS})
