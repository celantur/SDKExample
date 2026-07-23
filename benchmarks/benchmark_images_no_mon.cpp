#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "ThreadSafeQueue.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include <boost/program_options.hpp>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

// ============================================================
// USER CONFIGURATION
// Edit create_processor() to configure your benchmark setup.
// Do not modify main() below.
// ============================================================

std::unique_ptr<CelanturSDK::Processor> create_processor(int tiling) {
    const std::filesystem::path assets_path = std::filesystem::path(boost::dll::program_location().parent_path().string()) / ".." / ".." / "assets";
    const std::filesystem::path license  = assets_path / "license";
    const std::filesystem::path model    = assets_path / "v10-static-fp32-medium-1280.onnx.enc";
    const std::filesystem::path prebuilt = assets_path / "v10-static-fp32-medium-1280.trt.enc";
    const std::filesystem::path compiled = std::filesystem::exists(prebuilt)
        ? prebuilt : assets_path / "v10-static-fp32-medium-1280.trt";
    // const std::filesystem::path plugin   = "/usr/local/lib/libTensorRTRuntime.so";
    const std::filesystem::path plugin   = "/app/output/lib/libTensorRTRuntime.so";

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

    if (tiling == 2) {
        params.n_tiles_x = 2;
        params.n_tiles_y = 1;
        params.ol_x = 0.05f;
    }

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

struct ImageTask {
    std::filesystem::path path;
    cv::Mat image;
    bool sentinel = false;
};

void reader_worker(
    int worker_id,
    int n_workers,
    const std::vector<std::filesystem::path>& images,
    ThreadSafeQueue<ImageTask>& to_process)
{
    for (size_t i = static_cast<size_t>(worker_id); i < images.size(); i += static_cast<size_t>(n_workers)) {
        ImageTask task;
        task.path  = images[i];
        task.image = cv::imread(images[i].string());
        to_process.push(std::move(task));
    }

    ImageTask end;
    end.sentinel = true;
    to_process.push(std::move(end));
}

void processor_worker(
    ThreadSafeQueue<ImageTask>& to_process,
    ThreadSafeQueue<ImageTask>& to_write,
    CelanturSDK::Processor& processor)
{
    while (true) {
        ImageTask task = to_process.pop();
        if (task.sentinel) {
            to_write.push(std::move(task));
            break;
        }

        processor.process(std::move(task.image));
        task.image = processor.get_result();
        processor.get_detections();
        to_write.push(std::move(task));
    }
}

void writer_worker(
    ThreadSafeQueue<ImageTask>& to_write,
    const std::filesystem::path& output_dir)
{
    while (true) {
        ImageTask task = to_write.pop();
        if (task.sentinel)
            break;

        if (!output_dir.empty())
            cv::imwrite((output_dir / task.path.filename()).string(), task.image);
    }
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h",   "Show this help message")
        ("input,i",  po::value<std::string>()->required(),          "Directory of images to process")
        ("output,o", po::value<std::string>()->default_value(""),   "Directory to write anonymised images (optional)")
        ("n-proc,j", po::value<int>()->default_value(1),             "Number of reader/processor/writer pipelines")
        ("tiling,t", po::value<int>()->default_value(1),              "Tiling preset (1=none, 2=2x1 tiles with 5% overlap)");

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
    const int tiling             = vm["tiling"].as<int>();

    if (n_proc < 1) {
        std::cerr << "Error: n-proc must be at least 1\n";
        return 1;
    }

    if (tiling != 1 && tiling != 2) {
        std::cerr << "Error: tiling must be 1 or 2\n";
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
    for (int i = 0; i < n_proc; ++i)
        processors.push_back(create_processor(tiling));

    const size_t queue_depth = static_cast<size_t>(n_proc) * 2;

    std::deque<ThreadSafeQueue<ImageTask>> to_process;
    std::deque<ThreadSafeQueue<ImageTask>> to_write;
    for (int i = 0; i < n_proc; ++i) {
        to_process.emplace_back(queue_depth);
        to_write.emplace_back(queue_depth);
    }

    std::cout << "Processors ready. Starting pipelines...\n";

    std::vector<std::thread> readers;
    std::vector<std::thread> processors_threads;
    std::vector<std::thread> writers;
    readers.reserve(n_proc);
    processors_threads.reserve(n_proc);
    writers.reserve(n_proc);

    for (int i = 0; i < n_proc; ++i) {
        readers.emplace_back(reader_worker, i, n_proc, std::cref(images), std::ref(to_process[i]));
        processors_threads.emplace_back(
            processor_worker, std::ref(to_process[i]), std::ref(to_write[i]), std::ref(*processors[i]));
        writers.emplace_back(writer_worker, std::ref(to_write[i]), output_dir);
    }

    for (auto& t : readers)
        t.join();
    for (auto& t : processors_threads)
        t.join();
    for (auto& t : writers)
        t.join();

    std::cout << "Done. Processed " << images.size() << " images.\n";
    return 0;
}
