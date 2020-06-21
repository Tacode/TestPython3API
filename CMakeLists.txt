cmake_minimum_required(VERSION 3.5)
set(OpenCV_DIR "/usr/local/opencv2.4.9/share/OpenCV")
project(TestPython3API LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 2.4.9 QUIET)

find_package(Eigen3 3.1.0 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/include
                    ${EIGEN3_INCLUDE_DIR}
                    /usr/include/python3.5
                    /usr/lib/python3/dist-packages/numpy/core/include/numpy
                    )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
src/conversion.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    /usr/lib/x86_64-linux-gnu/libpython3.5m.so
    )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(TestBase testBase.cpp)
target_link_libraries(TestBase
                      /usr/lib/x86_64-linux-gnu/libboost_python-py35.so
                      /usr/lib/x86_64-linux-gnu/libstdc++.so.6
                      /usr/lib/x86_64-linux-gnu/libpython3.5m.so
                      ${PROJECT_NAME}
                      )

add_executable(TestMat testMat.cpp)
target_link_libraries(TestMat
                      /usr/lib/x86_64-linux-gnu/libboost_python-py35.so
                      /usr/lib/x86_64-linux-gnu/libstdc++.so.6
                      /usr/lib/x86_64-linux-gnu/libpython3.5m.so
                      ${PROJECT_NAME}
                      )
