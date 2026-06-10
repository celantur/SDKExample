#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/dll.hpp>
#include <opencv2/opencv.hpp>

#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "ImageEncoding.h"

/**
    Shared boilerplate for the Celantur SDK examples.

    Every example needs the same set of paths (license, input image, assets and output
    directories) and the same handful of inference plugin locations. On top of that they share
    the same processor configuration and the same "process one image and save it" sequence.
    Instead of repeating that boilerplate in every file, it lives here once. The backend-specific
    SDK usage (model compilation, inference settings, GPU handling, tracking, ...) is kept inline
    in each example so they remain readable as standalone tutorials.
 */
namespace example {

// Directory layout: executables live in build/, while assets/ and output/ sit next to it.
inline const std::filesystem::path exe_path    = boost::dll::program_location().parent_path().string();
inline const std::filesystem::path assets_path = exe_path / ".." / "assets";
inline const std::filesystem::path output_path = exe_path / ".." / "output";

// Assets shared by (almost) every example.
inline const std::filesystem::path license_file = assets_path / "license";
inline const std::filesystem::path image_path   = assets_path / "image.jpg";
inline const std::filesystem::path model_path   = assets_path / "v10-static-fp32-medium-1280.onnx.enc";

// Inference plugins shipped with the SDK.
inline const std::filesystem::path onnx_plugin     = "/usr/local/lib/libONNXInference.so";
inline const std::filesystem::path openvino_plugin = "/usr/local/lib/libOpenVINORuntime.so";
inline const std::filesystem::path tensorrt_plugin = "/usr/local/lib/libTensorRTRuntime.so";

// Resolve an asset path by file name, e.g. example::asset("v6-static-fp32.onnx.enc").
inline std::filesystem::path asset(const std::string& name) {
    return assets_path / name;
}

// Resolve an output path, creating the output directory on first use.
inline std::filesystem::path output_file(const std::string& name) {
    std::filesystem::create_directories(output_path);
    return output_path / name;
}

// Default processor configuration shared by the examples. Pass the inference plugin to use
// (e.g. example::onnx_plugin): tiling with overlap, lowered thresholds for small classes, and a
// per-object-type processing config that fully blurs people, rectangle-blurs license plates, and
// detects (without blurring) vehicles and faces. See quickstart.cpp for a worked example.
inline celantur::ProcessorParams make_processor_params(const std::filesystem::path& inference_plugin) {
    celantur::ProcessorParams params;

    // Point the processor at the requested inference plugin.
    params.inference_plugin = inference_plugin;

    // OpenCV uses BGR by default, but the Celantur SDK uses RGB, so we need to set swapRB to true.
    params.swapRB = true;

    // Split the image into tiles; gives more precision but slower processing.
    params.n_tiles_x = 2;
    params.n_tiles_y = 2;

    // Overlap tiles by 10% to make detection more robust in the crossection area between tiles.
    params.ol_x = 0.1;
    params.ol_y = 0.1;

    // Process the whole image. Rectangle format is x, y, width, height; the sum of x and width
    // should be less or equal to 1, the same goes for y and height.
    params.roi = cv::Rect_<float>(0.0f, 0.0f, 1.0f, 1.0f);

    // Lower thresholds for the small/hard classes.
    celantur::ModelThresholds thresholds;
    thresholds.face = 0.1f;
    thresholds.license_plate = 0.1f;
    params.thresholds = thresholds;

    // Configure per-object-type processing. Each DetectionProcessingConfig is initialised positionally:
    //   { blur_type, gradient_start, gradient_stop, kernel_size, beautify_blur, detection_type }
    // Use BlurType::None together with a DetectionType to detect a class without anonymising it.
    params.per_type_processing_config = celantur::PerTypeProcessingConfig(
        {
            // Person: fully blurred using the segmentation mask.
            {
                celantur::CelanturClassId::Person,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::None,
                    0.3f, 0.0f, 0.2f, false,
                    celantur::DetectionType::Segmentation,
                }
            },
            // License plate: blurred with a rectangular shape.
            {
                celantur::CelanturClassId::LicensePlate,
                celantur::DetectionProcessingConfig {
                    celantur::BlurType::BBox_Rectangle,
                    0.3f, 0.0f, 1.0f, false,
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
                    celantur::BlurType::BBox_Oval,
                    0.3f, 0.0f, 0.2f, true,
                    celantur::DetectionType::BBox,
                }
            },
        }
    );

    return params;
}

// Run the processor over the shared example image and save the anonymised result under output_name.
// Uses OpenCV to load and save the image (BGR<->RGB is handled by swapRB in the processor params).
inline void process_image(CelanturSDK::Processor& processor, const std::string& output_name) {
    // Load some image for processing.
    std::cout << "loading image from " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);

    // Enqueue the image for processing and fetch the result.
    processor.process(image);
    cv::Mat out = processor.get_result();

    // Discard the detections. Necessary to free up the memory.
    processor.get_detections();

    // Save the result.
    const std::filesystem::path out_image_path = output_file(output_name);
    std::cout << "saving result to " << out_image_path << std::endl;
    cv::imwrite(out_image_path, out);
}

} // namespace example
