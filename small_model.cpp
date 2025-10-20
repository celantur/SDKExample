#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path cpu_plugin_location = "/usr/local/lib/libOpenVINORuntime.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"openvino_small_model.jpg"; 
const std::filesystem::path model_path = assets_path/"v6-static-fp32-small-640.onnx.enc"; 
const std::filesystem::path model_path_compiled = assets_path/"v6-static-fp32-small-640.openvino";

/**
    The purpose of this example is to show how to use SDK with some smaller models that can run on CPU and give some performance benefits
 */

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);

    // First, compile openvino model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = cpu_plugin_location;
        CelanturSDK::ModelCompiler compiler(license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);
       
        // Optionally, investigate/change something if you want to
        std::optional<int> num_threads = std::any_cast<std::optional<int> >(settings["num_threads"]);
        if (num_threads) {
            std::cout << "Automatically detected number of threads: " << *num_threads << std::endl;
        } else {
            std::cout << "Could not automatically detect number of threads, letting openvino to decide number of threads" << std::endl;
        }

        // We can still manually set the number of threads if we want to
        settings["num_threads"] = std::optional<int>(1); // Uncomment to set number of threads to 1

        // Now compile the model
        std::cout << "Compiling model to " << model_path_compiled << std::endl;
        compiler.compile_model(settings, model_path_compiled);
    }

    // Second, start the processor with the compiled model

    celantur::ProcessorParams params;
    // Manually point to the CPU inference plugin
    params.inference_plugin = cpu_plugin_location;
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

    // Set additional processor parameters to take care of smaller model size; If using model size 640, set context size to 640x640
    CelanturSDK::AdditionalProcessorParams additional_params;
    additional_params.context_height = 640;
    additional_params.context_width = 640;

    // Load the compiled inference model.
    std::cout << "load model from " << model_path << std::endl;
    processor.load_inference_model(settings, additional_params);

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

