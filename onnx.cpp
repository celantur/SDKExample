#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "example_common.h"
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show the options one can tinker with using CPU inference engine.
    Some of the settings might be internal and therefore not documented.
 */

int main(int argc, char** argv) {
    // Build the shared processor configuration for the CPU (ONNX) inference plugin
    celantur::ProcessorParams params = example::make_processor_params(example::onnx_plugin);
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Get the available inference engine settings and their default values
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(example::model_path);
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
    std::cout << "load model from " << example::model_path << std::endl;
    processor.load_inference_model(settings);

    // Process the shared example image and save the result
    example::process_image(processor, "onnx.jpg");

    return 0;
}
