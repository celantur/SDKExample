#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "ImageEncoding.h"
#include "example_common.h"
#include <fstream>
#include <opencv2/opencv.hpp>

/**
    The purpose of this example is to show the quickest and easiest way on how to use the CelanturSDK to anonymise an image, running everything on the CPU.

    It demonstrates two ways of getting images in and out of the SDK:
      1. Using OpenCV to load and save images. Note that OpenCV loads images by default in the BGR
         format, while the Celantur SDK uses RGB (hence swapRB).
      2. Using the SDK's own JPEG decode/encode functions, which also let you preserve the original
         EXIF metadata and control the encoding quality.
 */

int main(int argc, char** argv) {
    celantur::ProcessorParams params;

    // Manually point to the CPU inference plugin
    params.inference_plugin = example::onnx_plugin;
    std::cout << "Looking for license at " << example::license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true;

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, example::license_file);

    // Load the inference model. Should be provided by Celantur
    std::cout << "load model from " << example::model_path << std::endl;
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(example::model_path);
    processor.load_inference_model(settings);

    // ---------------------------------------------------------------------------------------------
    // Version 1: the simplest path, using OpenCV to load and save the image. (Disabled by default.)
    // ---------------------------------------------------------------------------------------------
    /*
    // Load some image for processing
    std::cout << "loading image from " << example::image_path << std::endl;
    cv::Mat image = cv::imread(example::image_path);

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();

    // Discard the detections. Necessary to free up the memory.
    processor.get_detections();

    // Save the result
    const std::filesystem::path out_image_path = example::output_file("quickstart.jpg");
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, out);
    */

    // ---------------------------------------------------------------------------------------------
    // Version 2: using the SDK's own JPEG decode/encode, preserving EXIF metadata.
    // We avoid using OpenCV directly here, as the SDK provides its own image encoding and decoding
    // functions. The SDK also allows to retain the image metadata and set the encoding quality.
    // ---------------------------------------------------------------------------------------------

    // Load the image binary
    std::ifstream image_file(example::image_path, std::ios::binary);
    std::vector<unsigned char> image_data((std::istreambuf_iterator<char>(image_file)), std::istreambuf_iterator<char>());

    // Decode the image
    cv::Mat decoded_image = celantur::jpeg_decode(image_data.data(), image_data.size());

    // Extract exif metadata from the image
    celantur::ExifMetadata metadata = celantur::jpeg_get_exif_metadata(image_data.data(), image_data.size());

    // Print the metadata
    metadata.print_debug_info(std::cout);

    // Enqueue the image for processing
    processor.process(decoded_image);

    // Get the result
    cv::Mat decoded_out = processor.get_result();

    // Discard the detections. Necessary to free up the memory.
    processor.get_detections();

    // Encode the image back together with metadata. Notice the std::move to transfer the ownership of the metadata to the encoder. We are using thin wrapper around exiflib to handle the metadata and there is no copy constructor defined
    std::vector<unsigned char> encoded_image = celantur::jpeg_encode(decoded_out, 95, std::move(metadata));

    // Write to the file
    const std::filesystem::path out_metadata_image_path = example::output_file("quickstart_with_metadata.jpg");
    std::cout << "saving result with metadata to " << out_metadata_image_path << std::endl;
    std::ofstream output_image(out_metadata_image_path, std::ios::binary);
    output_image.write(reinterpret_cast<const char*>(encoded_image.data()), encoded_image.size());

    return 0;
}
