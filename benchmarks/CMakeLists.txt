add_executable(benchmark_images benchmark_images.cpp)
target_link_libraries(benchmark_images PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module Boost::program_options ${CMAKE_DL_LIBS})

add_executable(benchmark_images_no_mon benchmark_images_no_mon.cpp)
target_link_libraries(benchmark_images_no_mon PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module Boost::program_options ${CMAKE_DL_LIBS})

add_executable(benchmark_video benchmark_video.cpp)
target_link_libraries(benchmark_video PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module Boost::program_options ${CMAKE_DL_LIBS})
