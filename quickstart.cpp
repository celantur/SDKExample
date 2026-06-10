#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "ImageEncoding.h"
#include "detections-utils.h"
#include "example_common.h"
#include <fstream>
#include <map>
#include <sstream>
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show how to use the CelanturSDK to anonymise an image on the CPU,
    from the basic setup all the way to configuring the processor in detail and inspecting the results.

    It combines several aspects:
      - Configuring processor parameters: tiling, overlap, region-of-interest (ROI), per-class
        thresholds and per-object-type processing. The shared defaults live in
        example::make_processor_params (example_common.h).
      - Per-object-type processing via DetectionProcessingConfig, showcasing every BlurType / DetectionType:
          * Person:        fully blurred using the segmentation mask.
          * License plate: blurred with a rectangular shape.
          * Vehicle:       detected only (no blur) using a segmentation mask; the count is printed.
          * Face:          detected only (no blur) using a bounding box; the count is printed.
      - Visualising detections and serialising metrics for debugging.
      - Two ways of getting images in and out of the SDK:
          1. Using OpenCV to load and save images. Note that OpenCV loads images by default in the BGR
             format, while the Celantur SDK uses RGB (hence swapRB).
          2. Using the SDK's own JPEG decode/encode functions, which also let you preserve the original
             EXIF metadata and control the encoding quality.
 */

int main(int argc, char** argv) {
    // Build the shared processor configuration (tiling, overlap, ROI, thresholds and per-object-type
    // processing config). See example::make_processor_params in example_common.h for the details.
    celantur::ProcessorParams params = example::make_processor_params(example::onnx_plugin);

    std::cout << "Looking for license at " << example::license_file << std::endl;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Load the inference model. Should be provided by Celantur
    std::cout << "load model from " << example::model_path << std::endl;
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(example::model_path);
    processor.load_inference_model(settings);

    // ---------------------------------------------------------------------------------------------
    // Version 1: OpenCV to load and save the image, with detection visualisation and metric output.
    // ---------------------------------------------------------------------------------------------

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

    // Count detections per class (handy for the detect-only classes such as vehicles and faces)
    std::map<int, int> class_counts;
    for (const celantur::CelanturDetection& det : dets) {
        class_counts[det.class_id]++;
    }
    std::cout << "Detected vehicles: " << class_counts[static_cast<int>(celantur::CelanturClassId::Vehicle)] << std::endl;
    std::cout << "Detected faces: " << class_counts[static_cast<int>(celantur::CelanturClassId::Face)] << std::endl;

    // output metrics to file
    std::ofstream metadata_json_file(example::output_file("metadata.json"));
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_file);

    // save metrics as a string
    std::stringstream metadata_json_str;
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_str);
    std::cout << "metadata json: " << metadata_json_str.str() << std::endl;

    // Save the result
    const std::filesystem::path out_image_path = example::output_file("quickstart.jpg");
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, result);

    // ---------------------------------------------------------------------------------------------
    // Version 2: using the SDK's own JPEG decode/encode, preserving EXIF metadata.
    // We avoid using OpenCV directly here, as the SDK provides its own image encoding and decoding
    // functions. The SDK also allows to retain the image metadata and set the encoding quality.
    // ---------------------------------------------------------------------------------------------
    example::process_image_with_metadata(processor, "quickstart_with_metadata.jpg");

    return 0;
}
