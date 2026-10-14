cmake_minimum_required(VERSION 3.10)
Project(CelanturSDKExample LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS json log program_options REQUIRED)
find_package(CppProcessing REQUIRED CelanturSDK common-module)

message("Cpp Processing libraries found: ${CppProcessing_LIBRARIES}")

add_executable(onnx onnx.cpp)
target_link_libraries(onnx PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(jpeg jpeg.cpp)
target_link_libraries(jpeg PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(openvino openvino.cpp)
target_link_libraries(openvino PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(tensorrt tensorrt.cpp)
target_link_libraries(tensorrt PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(tracking tracking.cpp)
target_link_libraries(tracking PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

# Video anonymisation with FFmpeg decode/encode (CPU libx265 and/or NVIDIA hevc_cuvid/hevc_nvenc).
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)
add_executable(video_ffmpeg video_ffmpeg.cpp)
target_link_libraries(video_ffmpeg
    PRIVATE
        CppProcessing::CelanturSDK
        CppProcessing::common-module
        Boost::program_options
        PkgConfig::FFMPEG
)

# Video anonymisation with OpenCV decode/encode (codec backend via OpenCV/FFmpeg / env vars).
add_executable(video_opencv video_opencv.cpp)
target_link_libraries(video_opencv
    PRIVATE
        CppProcessing::CelanturSDK
        CppProcessing::common-module
        Boost::program_options
)

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
