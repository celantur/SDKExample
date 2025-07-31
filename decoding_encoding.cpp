#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include "ImageEncoding.h"

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path cpu_plugin_location = "/usr/local/lib/libONNXInference.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"decoding_encoding.jpg"; 
const std::filesystem::path model_path = assets_path/"yolov8_all_1280_medium_v6_static.onnx.enc"; 

/**
    This example expands on the quickstart example by showing how to encode and decode the images using the CelanturSDK.
    We avoid using OpenCV directly here, as the SDK provides its own image encoding and decoding functions.
    The SDK also allows to pertain the image metadata and set encoding quality.
 */

int main(int argc, char** argv) {
    std::filesystem::create_directories(output_path);
    celantur::ProcessorParams params;

    // Manually point to the CPU inference plugin
    params.inference_plugin = cpu_plugin_location;
    std::cout << "Looking for license at " << license_file << std::endl;

    // OpenCV uses by default BGR, but the Celantur SDK uses RGB so we need to set swapRB to true
    params.swapRB = true; 

    // Start the processor with given parameters and license file
    CelanturSDK::Processor processor(params, license_file);

    // Load the inference model. Should be provided by Celantur
    std::cout << "load model from " << model_path << std::endl;
    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path);
    processor.load_inference_model(settings);

    // Load the image binary
    std::ifstream image_file(image_path, std::ios::binary);
    std::vector<unsigned char> image_data((std::istreambuf_iterator<char>(image_file)), std::istreambuf_iterator<char>());
    
    // Decode the image
    cv::Mat image = celantur::jpeg_decode(image_data.data(), image_data.size());

    // Extract exif metadata fromt the image
    celantur::ExifMetadata metadata = celantur::jpeg_get_exif_metadata(image_data.data(), image_data.size());

    // Print the metadata
    metadata.print_debug_info(std::cout);   

    // Enqueue the image for processing
    processor.process(image);

    // Get the result
    cv::Mat out = processor.get_result();
    
    // Discard the detections. Necessary to free up the memory. 
    processor.get_detections();

    // Encode the image back together with metadata. Notice the std::move to transfer the ownership of the metadata to the encoder. We are using thin wrapper around exiflib to handle the metadata and there is no copy constructor defined
    std::vector<unsigned char> encoded_image = celantur::jpeg_encode(out, 95, std::move(metadata));

    // Write to the file
    std::ofstream output_image(out_image_path, std::ios::binary);
    output_image.write(reinterpret_cast<const char*>(encoded_image.data()), encoded_image.size());
    return 0;
}
