#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "CompilationParams.h"
#include "example_common.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <tuple>

const std::filesystem::path video_path = example::asset("video.mp4");
const std::filesystem::path model_path = example::asset("v8-static-fp32-small-640.onnx.enc");
const std::filesystem::path model_path_compiled = example::asset("v8-static-fp32-small-640.trt.enc");

/**
    The purpose of this example is to show how one can improve the detection results for videos with tracking
*/
void process_video_with_tracking(CelanturSDK::Processor& processor, CelanturSDK::Tracker& tracker, CelanturSDK::Anonymiser& anonymiser) {
    const std::filesystem::path out_video_path = example::output_file("video.mp4");
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
    // First, compile tensor rt model if it does not exist
    if (!std::filesystem::exists(model_path_compiled)) {
        // Preload model to get to the compilation settings
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = example::tensorrt_plugin;
        CelanturSDK::ModelCompiler compiler(example::license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);

        // TensorRT has some specific settings we can modify; for example, we can set the precision and optimisation level
        // As this is an example, we opt for low optimisation level and FP32 precision for fast compilation
        std::cout << "Model compiler parameters:" << std::endl;
        std::cout << settings << std::endl;
        settings["precision"] = celantur::CompilePrecision::FP32;
        settings["optimisation_level"] = celantur::OptimisationLevel::Low;
        settings["min_opt_max_dims"] = celantur::MinOptMaxDims{std::make_tuple(640,640,640)};

        // Now compile the model
        std::cout << "Compiling model to " << model_path_compiled << std::endl;
        compiler.compile_model(settings, model_path_compiled);
    }

    // Second, start the processor with the compiled model
    celantur::ProcessorParams params;
    params.n_tiles_x = 2;
    // Manually point to the GPU inference plugin
    params.inference_plugin = example::tensorrt_plugin;
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Get the available inference engine settings and their default values
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path_compiled);
    std::cout << "Inference engine parameters:" << std::endl;
    std::cout << settings << std::endl;

    // Load the compiled inference model.
    std::cout << "load model from " << model_path << std::endl;
    processor.load_inference_model(settings, CelanturSDK::AdditionalProcessorParams{640, 640});

    // To perform tracking we need separate anonymiser and tracker objects.
    CelanturSDK::Tracker tracker(example::license_file);
    CelanturSDK::Anonymiser anonymiser(example::license_file, params.per_type_processing_config);

    process_video_with_tracking(processor, tracker, anonymiser);

    return 0;
}
