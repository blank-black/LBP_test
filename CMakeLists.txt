cmake_minimum_required(VERSION 3.8)
project(test_LBP)
find_package( OpenCV )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

add_executable(test_LBP ${SOURCE_FILES})

target_link_libraries(test_LBP ${OpenCV_LIBS} )
