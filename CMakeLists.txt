cmake_minimum_required(VERSION 3.10)
Project(CelanturSDKExample LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS json log REQUIRED)
find_package(CppProcessing REQUIRED CelanturSDK common-module)

message("Cpp Processing libraries found: ${CppProcessing_LIBRARIES}")

add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(jpeg jpeg.cpp)
target_link_libraries(jpeg PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(openvino openvino.cpp)
target_link_libraries(openvino PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(tensorrt tensorrt.cpp)
target_link_libraries(tensorrt PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(tracking tracking.cpp)
target_link_libraries(tracking PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

# GPU inference
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    add_executable(cuda cuda.cpp)

    target_link_libraries(cuda
        PRIVATE
            CppProcessing::CelanturSDK
            CppProcessing::common-module
            CUDA::cudart
    )
    message(STATUS "CUDA found: cuda will be built.")
else()
    # 2. Output an error/warning message without stopping the build
    message(WARNING "CUDA NOT FOUND: cuda will be skipped, but other targets will still compile.")
endif()
