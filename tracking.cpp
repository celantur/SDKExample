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
const std::filesystem::path video_path = assets_path/"video.mp4";
const std::filesystem::path out_video_path = output_path/"video.mp4";
const std::filesystem::path model_path = assets_path/"v6-static-fp32.onnx.enc";
const std::filesystem::path model_path_compiled = assets_path/"v6-static-fp32-compiled.trt";

/**
    The purpose of this example is to show how one can improve the detection results for videos with tracking 
*/
void process_video_with_tracking(CelanturSDK::Processor& processor, CelanturSDK::Tracker& tracker, CelanturSDK::Anonymiser& anonymiser) {
    cv::VideoCapture cap(video_path.string());

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source." << std::endl;
        return;
    }

    // 2. Get properties from the source
    int frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps       = cap.get(cv::CAP_PROP_FPS);
    
    cv::VideoWriter writer(out_video_path.string(),
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps, 
                           cv::Size(frame_width, frame_height));

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open VideoWriter." << std::endl;
        return;
    }

    cv::Mat frame;
    int counter = 0;


    while (true) {
        // 4. Read frame
        bool success = cap.read(frame);
        if (!success || frame.empty()) {
            std::cout << "End of stream." << std::endl;
            break;
        }

        std::cout << "Processing frame " << ++counter << std::endl;

        processor.process(frame);
        cv::Mat anonymised_frame = processor.get_result();
        std::vector<celantur::CelanturDetection> dets = processor.get_detections();
        tracker.update(dets);

        auto lost_dets = tracker.get_lost_detections();
        for (const auto& det : lost_dets) {
            anonymiser.anonymise(anonymised_frame, det);
            // Just for visualisation purposes, draw the bounding box of the lost track on the frame
            cv::rectangle(anonymised_frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        // 5. Write frame to file
        writer.write(anonymised_frame);
    }

    // 7. Cleanup (Release handles automatically on destruction, but good practice)
    cap.release();
    writer.release();

    return;
}

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);

    // First, compile tensor rt model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = gpu_plugin_location;
        CelanturSDK::ModelCompiler compiler(license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);
       
        // TensorRT has some specific settings we can modify; for example, we can set the precision and optimisation level
        // As this is an example, we opt for low optimisation level and FP32 precision for fast compilation
        settings["precision"] = celantur::CompilePrecision::FP32;
        settings["optimisation_level"] = celantur::OptimisationLevel::Low;

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

    // To perform tracking we need separate anonymiser and tracker objects.
    CelanturSDK::Tracker tracker(license_file);
    CelanturSDK::Anonymiser anonymiser(license_file, params.per_type_processing_config);

    process_video_with_tracking(processor, tracker, anonymiser);

    return 0;
}



