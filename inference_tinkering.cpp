#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path cpu_plugin_location = "/usr/local/lib/libONNXInference.so";
const std::filesystem::path license_file = exe_path/"license";
const std::filesystem::path image_path = exe_path/"image.jpg";
const std::filesystem::path out_image_path = exe_path/"inference_tinkering.jpg"; 
const std::filesystem::path model_path = exe_path/"yolov8_all_1280_medium_v4_static.onnx.enc"; 

/**
    The purpose of this example is to show the options one can tinker with using CPU inference engine.
    Some of the settings might be internal and therefore not documented. 
 */

int main(int argc, char** argv) {
    celantur::ProcessorParams params;

    // Manually point to the CPU inference plugin
    params.inference_plugin = cpu_plugin_location;
    std::cout << "Looking for license at " << license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true; 

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, license_file);

    // Get the available inference engine settings and their default values
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings();
    std::cout << "Inference engine parameters:" << std::endl;
    for (const std::pair<std::string, std::any>& pair : settings) {
        std::cout << pair.first  << std::endl;
    }
    
    // set the number of inference threads to 1 to limit inference engine to work with only one thread; value of 0 means that the engine will use all available threads
    settings["n_intra_threads"] = 1;
    settings["n_outer_threads"] = 1;
    // set the optimisation level of the model to the highest (Already set by default)
    settings["optimisation_level"] = celantur::OptimisationLevel::Full;

    // Load the inference model. Should be provided by Celantur; the settings are provided to the load_inference_model function
    std::cout << "load model from " << model_path << std::endl;
    processor.load_inference_model(model_path, settings);

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