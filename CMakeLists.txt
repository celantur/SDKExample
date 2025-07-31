cmake_minimum_required(VERSION 3.10)
Project(CelanturSDKExample LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS json log REQUIRED)
find_package(CppProcessing REQUIRED CelanturSDK common-module) 

message("Cpp Processing libraries found: ${CppProcessing_LIBRARIES}")

add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(processor_parameters_detail processor_parameters_detail.cpp)
target_link_libraries(processor_parameters_detail PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(decoding_encoding decoding_encoding.cpp)
target_link_libraries(decoding_encoding PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(inference_tinkering_onnx inference_tinkering_onnx.cpp)
target_link_libraries(inference_tinkering_onnx PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(detections_and_thresholds detections_and_thresholds.cpp)
target_link_libraries(detections_and_thresholds PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)

add_executable(inference_tinkering_openvino inference_tinkering_openvino.cpp)
target_link_libraries(inference_tinkering_openvino PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)
