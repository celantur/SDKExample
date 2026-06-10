#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "example_common.h"
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show how to run the SDK using the Nvidia GPU inference engine TensorRT,
    including how to choose the compilation precision and optimisation level.

    TensorRT has some specific settings we can modify; for example the precision and optimisation level.
    These trade compilation time against runtime performance:
      - FP32 + Low  optimisation: fast to compile, good for quickly trying things out.
      - FP16 + Full optimisation: slow to compile, but best runtime performance.
    This example defaults to the fast FP32 + Low profile; uncomment the alternative for maximum performance.
 */

int main(int argc, char** argv) {
    const std::filesystem::path model_path_compiled = example::asset("v10-static-fp32-medium-1280.trt.enc");

    // First, compile tensor rt model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = example::tensorrt_plugin;
        CelanturSDK::ModelCompiler compiler(example::license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(example::model_path);

        // Fast-compiling profile: FP32 precision and Low optimisation level.
        settings["precision"] = celantur::CompilePrecision::FP32;
        settings["optimisation_level"] = celantur::OptimisationLevel::Low;

        // Maximum-performance profile: FP16 precision and Full optimisation level (slower to compile).
        // settings["precision"] = celantur::CompilePrecision::FP16;
        // settings["optimisation_level"] = celantur::OptimisationLevel::Full;

        // Now compile the model
        std::cout << "Compiling model to " << model_path_compiled << std::endl;
        compiler.compile_model(settings, model_path_compiled);
    }

    // Second, start the processor with the compiled model
    celantur::ProcessorParams params = example::make_processor_params(example::tensorrt_plugin);
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Get the available inference engine settings and their default values
    // For the tensorRT plugin currently there are no additional settings but this may change in the future
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path_compiled);
    std::cout << "Inference engine parameters:" << std::endl;
    for (const std::pair<std::string, std::any>& pair : settings) {
        std::cout << pair.first  << std::endl;
    }

    // Load the compiled inference model.
    std::cout << "load model from " << example::model_path << std::endl;
    processor.load_inference_model(settings);

    // Process the shared example image and save the result
    example::process_image(processor, "tensorrt.jpg");

    return 0;
}
