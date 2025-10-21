#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path gpu_plugin_location = "/usr/local/lib/libTensorRTRuntime.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"gpu_quickstart.jpg";
const std::filesystem::path model_path = assets_path/"v6-static-fp32.onnx.enc";
const std::filesystem::path model_path_compiled = assets_path/"v6-static-fp32-compiled.trt";

/**
    The purpose of this example is to show the options one can run SDK using Nvidia GPU inference engine TensorRT.
 */



    // PRINT_TEST_NAME();
    // CelanturSDK::ModelCompilerParams params;
    // params.inference_plugin = plugin_dir/trt_name;
    // CelanturSDK::ModelCompiler compiler(license_file, params);
    //
    // celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);
    // settings["precision"] = celantur::CompilePrecision::FP32;
    // settings["optimisation_level"] = celantur::OptimisationLevel::Low;
    // compiler.compile_model(settings, trt_comp_model);
    //
    // celantur::ProcessorParams proc_params;
    // proc_params.inference_plugin = plugin_dir/trt_name;
    // CelanturSDK::Processor processor(proc_params, license_file);
    // auto runtime_settings = processor.get_inference_settings(trt_comp_model);
    // CelanturSDK::AdditionalProcessorParams additional_params;
    // processor.load_inference_model(runtime_settings, additional_params);
    // anonymise_image(image_path, out_test/"trt_compile_run.jpg", processor);

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);

    // First, compile openvino model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = gpu_plugin_location;
        CelanturSDK::ModelCompiler compiler(license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);

        // Now compile the model
        std::cout << "Compiling model to " << model_path_compiled << std::endl;
        compiler.compile_model(settings, model_path_compiled);
    }

    // Second, start the processor with the compiled model
    celantur::ProcessorParams params;
    // Manually point to the CPU inference plugin
    params.inference_plugin = gpu_plugin_location;
    std::cout << "Looking for license at " << license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, license_file);

    // Get the available inference engine settings and their default values
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path_compiled);
    std::cout << "Inference engine parameters:" << std::endl;
    for (const std::pair<std::string, std::any>& pair : settings) {
        std::cout << pair.first  << std::endl;
    }

    // Load the compiled inference model.
    std::cout << "load model from " << model_path << std::endl;
    processor.load_inference_model(settings);

    // Load some image for processing
    std::cout << "loading image from " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();

    // Discard the detections. Necessary to free up the memory.
    processor.get_detections();

    // Save the result
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, out);

    return 0;
}


