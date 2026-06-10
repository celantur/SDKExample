#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "detections-utils.h"
#include "example_common.h"
#include <fstream>
#include <map>
#include <sstream>
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to dive into the parameters that can be set when creating a Processor object.
    It combines several aspects:
      - Processor parameters such as tiling and the region of interest (ROI) that the image is processed in.
      - Per-class detection thresholds.
      - Per-object-type processing via DetectionProcessingConfig, showcasing every BlurType / DetectionType:
          * Person:        fully blurred using the segmentation mask.
          * License plate: blurred with a rectangular shape.
          * Vehicle:       detected only (no blur) using a segmentation mask; the count is printed.
          * Face:          detected only (no blur) using a bounding box; the count is printed.
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

    // Configure per-object-type processing. Each DetectionProcessingConfig is initialised positionally:
    //   { blur_type, gradient_start, gradient_stop, kernel_size, beautify_blur, detection_type }
    // Use BlurType::None together with a DetectionType to detect a class without anonymising it.
    const celantur::PerTypeProcessingConfig anonymisation_config
    (
        {
            // Person: fully blurred using the segmentation mask.
            {
                celantur::CelanturClassId::Person,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::Segmentation,
                    0.3f, 0.0f, 0.2f, true,
                    celantur::DetectionType::Segmentation,
                }
            },
            // License plate: blurred with a rectangular shape.
            {
                celantur::CelanturClassId::LicensePlate,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::BBox_Rectangle,
                    0.3f, 0.0f, 1.0f, true,
                    celantur::DetectionType::BBox,
                }
            },
            // Vehicle: detected only (no blur) using a segmentation mask.
            {
                celantur::CelanturClassId::Vehicle,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::None,
                    0.3f, 0.0f, 0.2f, false,
                    celantur::DetectionType::Segmentation,
                }
            },
            // Face: detected only (no blur) using a bounding box.
            {
                celantur::CelanturClassId::Face,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::None,
                    0.3f, 0.0f, 0.2f, false,
                    celantur::DetectionType::BBox,
                }
            },
        }
    );

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
    const std::filesystem::path out_image_path = example::output_file("parametrisation.jpg");
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, result);

    return 0;
}
