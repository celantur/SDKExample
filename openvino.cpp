#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "example_common.h"
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show how to compile and run a model with the OpenVINO CPU inference engine,
    including inspecting/overriding compilation settings (e.g. thread count).

    By default it runs the full model. At the bottom of each relevant step you will find the commented-out
    "small model" variant: a smaller model (640) runs faster on CPU but requires setting the context size via
    AdditionalProcessorParams to match the model input size.
 */

int main(int argc, char** argv) {
    const std::filesystem::path model_path = example::asset("v6-static-fp32.onnx.enc");
    const std::filesystem::path model_path_compiled = example::asset("v6-static-fp32-compiled.openvino.enc");

    // Small model variant: faster on CPU, requires a matching context size (see below).
    // const std::filesystem::path model_path = example::asset("v8-static-fp32-small-640.onnx.enc");
    // const std::filesystem::path model_path_compiled = example::asset("v6-static-fp32-small-640.openvino");
    // constexpr int context_size = 640;

    // First, compile the OpenVINO model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = example::openvino_plugin;
        CelanturSDK::ModelCompiler compiler(example::license_file, compiler_params);
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
    params.inference_plugin = example::openvino_plugin;
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Get the available inference engine settings and their default values
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path_compiled);
    std::cout << "Inference engine parameters:" << std::endl;
    for (const std::pair<std::string, std::any>& pair : settings) {
        std::cout << pair.first  << std::endl;
    }

    // Load the compiled inference model.
    std::cout << "load model from " << model_path << std::endl;
    processor.load_inference_model(settings);

    // Small model variant: set the context size to match the model input size (640x640) and load
    // the model with these additional parameters instead of the plain load_inference_model above.
    // CelanturSDK::AdditionalProcessorParams additional_params;
    // additional_params.context_height = context_size;
    // additional_params.context_width = context_size;
    // processor.load_inference_model(settings, additional_params);

    // Load some image for processing
    std::cout << "loading image from " << example::image_path << std::endl;
    cv::Mat image = cv::imread(example::image_path);

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();

    // Discard the detections. Necessary to free up the memory.
    processor.get_detections();

    // Save the result
    const std::filesystem::path out_image_path = example::output_file("openvino.jpg");
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, out);

    return 0;
}
