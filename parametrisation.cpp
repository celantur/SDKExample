#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "detections-utils.h"
#include "example_common.h"
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to dive into the parameters that can be set when creating a Processor object.
    It combines several aspects:
      - Processor parameters such as tiling and the region of interest (ROI) that the image is processed in.
      - Per-class detection thresholds.
      - Per-object-type processing: which classes to anonymise and how (here, segmentation-blurring
        persons and vehicles while leaving faces and license plates untouched).
      - Visualising detections and serialising metrics for debugging.
 */

int main(int argc, char** argv) {
    const std::filesystem::path model_path = example::asset("v6-static-fp32.onnx.enc");

    celantur::ProcessorParams params;

    // Manually point to the CPU inference plugin
    params.inference_plugin = example::onnx_plugin;
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Set the number of tiles to split the image into; Gives more precision but slower processing
    params.n_tiles_x = 2;
    params.n_tiles_y = 2;

    // Set the overlap between tiles to 10% to make detection more robust in the crossection area between multiple tiles
    params.ol_x = 0.1;
    params.ol_y = 0.1;

    // Detect things only in the bottom right quadrant of the image
    // Rectangle format is x, y, width, height. The sum of x and width should be less or equal to 1, the same goes for y and height
    params.roi = cv::Rect_<float>(0.5f, 0.5f, 0.5f, 0.5f);

    // Setup different thresholds for different classes
    celantur::ModelThresholds thresholds;
    thresholds.face = 0.1f;
    thresholds.license_plate = 0.1f;
    params.thresholds = thresholds;

    // Configure per-object-type processing: segmentation-blur persons and vehicles,
    // while leaving faces and license plates untouched.
    const celantur::PerTypeProcessingConfig anonymisation_config
    (
        {
            {celantur::CelanturClassId::LicensePlate, {celantur::BlurType::None}},
            {celantur::CelanturClassId::Person, {celantur::BlurType::Segmentation}},
            {celantur::CelanturClassId::Face, {celantur::BlurType::None}},
            {celantur::CelanturClassId::Vehicle, {celantur::BlurType::Segmentation}},
        }
    );

    // Another option is to use predefined configurations:
    // const celantur::PerTypeProcessingConfig anonymisation_config = celantur::PERSON_VEHICLE_PROCESSING_CONFIG;

    params.per_type_processing_config = anonymisation_config;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Load the inference model. Should be provided by Celantur
    std::cout << "load model from " << model_path << std::endl;
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path);
    processor.load_inference_model(settings);

    // Load some image for processing
    std::cout << "loading image from " << example::image_path << std::endl;
    cv::Mat image = cv::imread(example::image_path);

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();

    // Get the detections and draw them on the image
    std::vector<celantur::CelanturDetection> dets = processor.get_detections();
    cv::Mat result = celantur::visualise_detections(out, dets);

    // output metrics to file
    std::ofstream metadata_json_file(example::output_file("metadata.json"));
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_file);

    // save metrics as a string
    std::stringstream metadata_json_str;
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_str);
    std::cout << "metadata json: " << metadata_json_str.str() << std::endl;

    // Save the result
    const std::filesystem::path out_image_path = example::output_file("parametrisation.jpg");
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, result);

    return 0;
}
