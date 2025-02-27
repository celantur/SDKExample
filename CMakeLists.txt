cmake_minimum_required(VERSION 3.10)
Project(CelanturSDKExample LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS json log REQUIRED)
find_package(CppProcessing REQUIRED CelanturSDK common-module) 

message("Cpp Processing libraries found: ${CppProcessing_LIBRARIES}")

# Add tests here
add_executable(quickstart quickstart.cpp)

#Link the example
target_link_libraries(quickstart PRIVATE CppProcessing::core CppProcessing::CelanturSDK ${OpenCV_LIBS})

install(TARGETS quickstart RUNTIME )