#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================
// USER CONFIGURATION
// Edit create_processor() to configure your benchmark setup.
// Do not modify main() below.
// ============================================================

std::unique_ptr<CelanturSDK::Processor> create_processor() {
    const std::filesystem::path assets_path = std::filesystem::path(boost::dll::program_location().parent_path().string()) / ".." / ".." / "assets";
    const std::filesystem::path license  = assets_path / "license";
    const std::filesystem::path model    = assets_path / "v10-static-fp32-medium-1280.onnx.enc";
    const std::filesystem::path prebuilt = assets_path / "v10-static-fp32-medium-1280-cuda.trt.enc";
    const std::filesystem::path compiled = std::filesystem::exists(prebuilt)
        ? prebuilt : assets_path / "v10-static-fp32-medium-1280.trt";
    const std::filesystem::path plugin   = "/usr/local/lib/libTensorRTRuntime.so";

    if (!std::filesystem::exists(compiled)) {
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = plugin;
        CelanturSDK::ModelCompiler compiler(license, compiler_params);
        celantur::InferenceEnginePluginCompileSettings compile_settings = compiler.preload_model(model);

        compile_settings["precision"] = celantur::CompilePrecision::FP32;
        compile_settings["optimisation_level"] = celantur::OptimisationLevel::Low;

        std::cout << "Compiling TensorRT model to " << compiled << "...\n";
        compiler.compile_model(compile_settings, compiled);
    }

    celantur::ProcessorParams params;
    params.inference_plugin = plugin;
    params.swapRB = true; // OpenCV loads BGR; SDK expects RGB

    // Optional: configure thresholds
    // params.thresholds.face = 0.5f;
    // params.thresholds.license_plate = 0.5f;

    // Optional: configure per-type anonymisation
    // params.per_type_processing_config[celantur::ObjectClass::Face].blur_strength = 15;

    auto processor = std::make_unique<CelanturSDK::Processor>(params, license);
    celantur::InferenceEnginePluginSettings settings = processor->get_inference_settings(compiled);

    // Optional: for small 640-resolution models, set context dimensions:
    // CelanturSDK::AdditionalProcessorParams additional;
    // additional.context_width  = 640;
    // additional.context_height = 640;
    // processor->load_inference_model(settings, additional);

    processor->load_inference_model(settings);
    return processor;
}

// ============================================================
// BENCHMARK LOGIC — do not modify below this line
// ============================================================

namespace {

namespace po = boost::program_options;

const std::vector<std::string> IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png"
};

bool is_image_file(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end();
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h",   "Show this help message")
        ("input,i",  po::value<std::string>()->required(),          "Directory of images to process")
        ("output,o", po::value<std::string>()->default_value(""),   "Directory to write anonymised images (optional)")
        ("n-proc,j", po::value<int>()->default_value(1),             "Number of parallel processors");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << "benchmark_images_no_mon: run image anonymisation without monitoring overhead\n\n"
                      << desc << "\n"
                      << "Edit create_processor() in the source file to configure the SDK.\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    const std::string input_str  = vm["input"].as<std::string>();
    const std::string output_str = vm["output"].as<std::string>();
    const int n_proc             = vm["n-proc"].as<int>();

    if (n_proc < 1) {
        std::cerr << "Error: n-proc must be at least 1\n";
        return 1;
    }

    const std::filesystem::path input_dir(input_str);
    if (!std::filesystem::is_directory(input_dir)) {
        std::cerr << "Error: not a directory: " << input_dir << "\n";
        return 1;
    }

    std::filesystem::path output_dir;
    if (!output_str.empty()) {
        output_dir = output_str;
        std::filesystem::create_directories(output_dir);
    }

    std::vector<std::filesystem::path> images;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir))
        if (entry.is_regular_file() && is_image_file(entry.path()))
            images.push_back(entry.path());
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        std::cerr << "No images found in: " << input_dir << "\n";
        return 1;
    }

    std::cout << "Found " << images.size() << " images. Setting up " << n_proc << " processor(s)...\n";
    std::vector<std::unique_ptr<CelanturSDK::Processor>> processors;
    processors.reserve(n_proc);
    for (int i = 0; i < n_proc; ++i)
        processors.push_back(create_processor());
    std::cout << "Processors ready. Processing in batches of " << n_proc << "...\n";

    for (size_t batch_start = 0; batch_start < images.size(); batch_start += n_proc) {
        const size_t batch_end  = std::min(batch_start + static_cast<size_t>(n_proc), images.size());
        const size_t batch_size = batch_end - batch_start;

        std::vector<cv::Mat> batch_images(batch_size);
        for (size_t i = 0; i < batch_size; ++i)
            batch_images[i] = cv::imread(images[batch_start + i].string());

        for (size_t i = 0; i < batch_size; ++i)
            processors[i]->process(batch_images[i]);

        for (size_t i = 0; i < batch_size; ++i) {
            cv::Mat out = processors[i]->get_result();
            processors[i]->get_detections();
            if (!output_dir.empty())
                cv::imwrite((output_dir / images[batch_start + i].filename()).string(), out);
        }
    }

    std::cout << "Done. Processed " << images.size() << " images.\n";
    return 0;
}
