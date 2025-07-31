#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path cpu_plugin_location = "/usr/local/lib/libONNXInference.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"segment-person-vehicle.jpg"; 
const std::filesystem::path model_path = assets_path/"yolov8_all_1280_medium_v6_static.onnx.enc"; 


/**
    The purpose of this example is to dive more into the parameters that can be set when creating a Processor object.
    These are responsible for the tiling and for the region of interest (ROI) that the image will be processed in.
 */

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);
    celantur::ProcessorParams params;
    
    // Manually point to the CPU inference plugin
    params.inference_plugin = cpu_plugin_location;
    std::cout << "Looking for license at " << license_file << std::endl;

    const celantur::PerTypeProcessingConfig anonymisation_config
    (
        {
            {celantur::CelanturClassId::LicensePlate, {celantur::BlurType::None}},
            {celantur::CelanturClassId::Person, {celantur::BlurType::Segmentation}},
            {celantur::CelanturClassId::Face, {celantur::BlurType::None}},
            {celantur::CelanturClassId::Vehicle, {celantur::BlurType::Segmentation}},
        }
    );

    params.per_type_processing_config = anonymisation_config;
    params.swapRB = true; // OpenCV uses by default BGR, but the Celantur SDK uses RGB

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

    // Discard the detections. Necessary to free up the memory. 
    processor.get_detections();

    // Save the result
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, out);

    return 0;
}

