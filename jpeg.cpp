#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "ImageEncoding.h"
#include "detections-utils.h"
#include "example_common.h"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show a complete CPU anonymisation workflow that goes beyond the
    basics, focusing on getting images in and out of the SDK while inspecting the detections.

    It combines several aspects:
      - The shared processor configuration from example::make_processor_params (example_common.h),
        which also documents the per-object-type processing used here.
      - Using the SDK's own JPEG decode/encode functions instead of OpenCV. These let us preserve the
        original EXIF metadata and control the encoding quality.
      - Inspecting the result: counting every detected object class, serialising the per-image metrics
        for debugging, and drawing the detections onto the image.

    The example writes a single JPEG that has the detections drawn on top of the anonymised image, with
    the original EXIF metadata preserved.
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

    // We use the SDK's own JPEG decode/encode functions instead of OpenCV directly, as they let us
    // preserve the original EXIF metadata and control the encoding quality.

    // Load the image binary
    std::cout << "loading image from " << example::image_path << std::endl;
    std::ifstream image_file(example::image_path, std::ios::binary);
    std::vector<unsigned char> image_data((std::istreambuf_iterator<char>(image_file)), std::istreambuf_iterator<char>());

    // Decode the image
    cv::Mat decoded_image = celantur::jpeg_decode(image_data.data(), image_data.size());

    // Extract exif metadata from the image and print it
    celantur::ExifMetadata metadata = celantur::jpeg_get_exif_metadata(image_data.data(), image_data.size());
    metadata.print_debug_info(std::cout);

    // Enqueue the image for processing, then fetch the result and the detections
    processor.process(decoded_image);
    cv::Mat out = processor.get_result();
    std::vector<celantur::CelanturDetection> dets = processor.get_detections();

    // Count and print every detected object class
    std::map<std::string, int> class_counts;
    for (const celantur::CelanturDetection& det : dets) {
        class_counts[det.class_name]++;
    }
    for (const std::pair<const std::string, int>& entry : class_counts) {
        std::cout << "Detected " << entry.first << ": " << entry.second << std::endl;
    }

    // Serialise the per-image metrics to a file
    std::ofstream metadata_json_file(example::output_file("metadata.json"));
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_file);

    // Serialise the per-image metrics to a string
    std::stringstream metadata_json_str;
    serialise_image_metrics_to_json(out, dets, "input_image_name", "input_folder", metadata_json_str);
    std::cout << "metadata json: " << metadata_json_str.str() << std::endl;

    // Draw the detections on top of the anonymised image
    cv::Mat visualised = celantur::visualise_detections(out, dets);

    // Encode the image back to JPEG together with the EXIF metadata, at quality 95. Notice the std::move
    // to transfer ownership of the metadata to the encoder: we use a thin wrapper around exiflib to
    // handle the metadata and there is no copy constructor defined.
    std::vector<unsigned char> encoded_image = celantur::jpeg_encode(visualised, 95, std::move(metadata));

    // Write the result (detections drawn, original EXIF metadata preserved) to file
    const std::filesystem::path out_image_path = example::output_file("jpeg_with_metadata.jpg");
    std::cout << "saving visualised result with metadata to " << out_image_path << std::endl;
    std::ofstream output_image(out_image_path, std::ios::binary);
    output_image.write(reinterpret_cast<const char*>(encoded_image.data()), encoded_image.size());

    return 0;
}
