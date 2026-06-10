#pragma once

#include <filesystem>
#include <string>
#include <boost/dll.hpp>

/**
    Shared boilerplate for the Celantur SDK examples.

    Every example needs the same set of paths (license, input image, assets and output
    directories) and the same handful of inference plugin locations. Instead of repeating
    that "heading" in every file, it lives here once. The actual SDK usage is kept inline
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

} // namespace example
