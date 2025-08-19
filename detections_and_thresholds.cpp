#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include <sstream>
#include "detections-utils.h"

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path cpu_plugin_location = "/usr/local/lib/libONNXInference.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"detections_and_thresholds.jpg";
const std::filesystem::path model_path = assets_path/"v6-static-fp32.onnx.enc";

/**
    This example expands on the quickstart example by showing how to set different thresholds for different classes.
    Also, it shows an example how to visualise detections for better debugging
 */

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);
    celantur::ProcessorParams params;

    // Manually point to the CPU inference plugin
    params.inference_plugin = cpu_plugin_location;
    std::cout << "Looking for license at " << license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Setup different thresholds for different classes
    celantur::ModelThresholds thresholds;
    thresholds.face = 0.1f;
    thresholds.license_plate = 0.1f;
    params.thresholds = thresholds;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, license_file);

    // Load the inference model. Should be provided by Celantur
    std::cout << "load model from " << model_path << std::endl;
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path);
    processor.load_inference_model(settings);

    // Load some image for processing
    std::cout << "loading image from " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();

    // Get the detections and draw them on the image
    std::vector<celantur::CelanturDetection> dets = processor.get_detections();
    cv::Mat result = celantur::visualise_detections(out, dets);

    // output metrics to file
    std::ofstream metadata_json_file(exe_path/"metadata.json");
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_file);

    // save metrics as a string
    std::stringstream metadata_json_str;
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_str);
    std::cout << "metadata json: " << metadata_json_str.str() << std::endl;




    // Save the result
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, result);

    return 0;
}
