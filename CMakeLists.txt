cmake_minimum_required(VERSION 3.10)
project(celantur_sdk_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED )

# If can't find the package, add the folliwind definition to CMAKE:
# -DCppProcessing_DIR=/usr/local/lib/cmake or the path where the package is installed
find_package(CppProcessing REQUIRED)

# In case there is some incompatibily between versions of OpenCV and Eigen, add the following definition to CMAKE
add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)


add_executable(celantur_sdk_example main.cpp)
target_link_libraries(celantur_sdk_example CppProcessing::processing ${OpenCV_LIBS})

install(TARGETS celantur_sdk_example DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/test-img.jpg DESTINATION .)



