#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include <boost/program_options.hpp>
#include "resource_monitor.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
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
    const std::filesystem::path compiled = assets_path / "v10-static-fp32-medium-1280.openvino";
    const std::filesystem::path plugin   = "/usr/local/lib/libOpenVINORuntime.so";

    if (!std::filesystem::exists(compiled)) {
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = plugin;
        CelanturSDK::ModelCompiler compiler(license, compiler_params);
        celantur::InferenceEnginePluginCompileSettings compile_settings = compiler.preload_model(model);

        // Optional: override thread count (default: auto-detected)
        // compile_settings["num_threads"] = std::optional<int>(4);

        std::cout << "Compiling OpenVINO model to " << compiled << "...\n";
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

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

const std::vector<std::string> IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png"
};

bool is_image_file(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end();
}

struct Stats { double total, mean, stddev, min, max; };

Stats compute_stats(const std::vector<double>& v) {
    double sum  = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq   = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return { sum, mean, std::sqrt(sq / v.size()),
             *std::min_element(v.begin(), v.end()),
             *std::max_element(v.begin(), v.end()) };
}

void print_stats_row(const std::string& label, const Stats& s) {
    std::cout << "  " << std::left  << std::setw(10) << label
              << std::right << std::fixed << std::setprecision(2)
              << "  mean " << std::setw(8) << s.mean  << " ms"
              << "   std " << std::setw(7) << s.stddev
              << "   min " << std::setw(7) << s.min
              << "   max " << std::setw(7) << s.max
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h",   "Show this help message")
        ("input,i",  po::value<std::string>()->required(),          "Directory of images to process")
        ("output,o", po::value<std::string>()->default_value(""),   "Directory to write anonymised images (optional)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << "benchmark_images: measure end-to-end image anonymisation performance\n\n"
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

    std::cout << "Found " << images.size() << " images. Setting up processor...\n";
    auto processor = create_processor();
    std::cout << "Processor ready. Starting benchmark...\n\n";

    std::vector<double> load_times, process_times, save_times, megapixels, widths, heights;
    std::vector<double> gpu_util, gpu_mem_mb;
    load_times.reserve(images.size());
    process_times.reserve(images.size());

    GpuMonitor gpu_monitor;
    if (gpu_monitor.available())
        std::cout << "GPU monitoring available (NVML).\n";

    const auto cpu_start  = read_cpu();
    const auto mem_start  = read_mem();
    const auto wall_start = Clock::now();

    for (const auto& img_path : images) {
        const auto t0 = Clock::now();
        cv::Mat image = cv::imread(img_path.string());
        const auto t1 = Clock::now();

        if (image.empty()) {
            std::cerr << "  Warning: failed to load " << img_path.filename() << ", skipping.\n";
            continue;
        }

        processor->process(image);
        cv::Mat out = processor->get_result();
        processor->get_detections();
        const auto t2 = Clock::now();

        const double load_ms    = elapsed_ms(t0, t1);
        const double process_ms = elapsed_ms(t1, t2);

        double save_ms = 0.0;
        if (!output_dir.empty()) {
            const auto t3 = Clock::now();
            cv::imwrite((output_dir / img_path.filename()).string(), out);
            save_ms = elapsed_ms(t3, Clock::now());
            save_times.push_back(save_ms);
        }

        if (auto s = gpu_monitor.sample()) {
            gpu_util.push_back(s->util_pct);
            gpu_mem_mb.push_back(s->proc_mem_mb);
        }

        load_times.push_back(load_ms);
        process_times.push_back(process_ms);
        megapixels.push_back(static_cast<double>(image.cols * image.rows) / 1e6);
        widths.push_back(image.cols);
        heights.push_back(image.rows);

        std::cout << std::left << std::setw(42) << img_path.filename().string()
                  << "  " << image.cols << "x" << image.rows
                  << "  load: "    << std::fixed << std::setprecision(1) << load_ms    << " ms"
                  << "  process: " << process_ms << " ms";
        if (!output_dir.empty())
            std::cout << "  save: " << save_ms << " ms";
        std::cout << "\n";
    }

    const double wall_ms  = elapsed_ms(wall_start, Clock::now());
    const auto   cpu_end  = read_cpu();
    const auto   mem_end  = read_mem();

    if (process_times.empty()) {
        std::cerr << "No images were successfully processed.\n";
        return 1;
    }

    const auto proc_s  = compute_stats(process_times);
    const auto load_s  = compute_stats(load_times);
    const auto mp_s    = compute_stats(megapixels);
    const auto w_s     = compute_stats(widths);
    const auto h_s     = compute_stats(heights);

    const double throughput_ips = static_cast<double>(process_times.size()) / (proc_s.total / 1000.0);
    const double throughput_mps = mp_s.total / (proc_s.total / 1000.0);

    std::cout << "\n"
              << "=========================================\n"
              << "           BENCHMARK RESULTS\n"
              << "=========================================\n"
              << std::fixed << std::setprecision(2)
              << "Images processed : " << process_times.size() << "\n"
              << "Wall-clock time  : " << wall_ms   / 1000.0 << " s\n"
              << "Total proc time  : " << proc_s.total / 1000.0 << " s\n"
              << "\n"
              << "Per-image timing:\n";
    print_stats_row("process", proc_s);
    print_stats_row("load",    load_s);
    if (!save_times.empty())
        print_stats_row("save", compute_stats(save_times));
    std::cout << "\n"
              << "Resolution:\n"
              << "  Avg             " << std::setprecision(0)
              << w_s.mean << "x" << h_s.mean << "\n"
              << std::setprecision(2)
              << "  Avg megapixels  " << mp_s.mean << " MP"
              << "  (min " << mp_s.min << "  max " << mp_s.max << ")\n"
              << "\n"
              << "Throughput (process time only):\n"
              << "  " << throughput_ips << " images/sec\n"
              << "  " << throughput_mps << " megapixels/sec\n";

    const CpuTime cpu_used     = { cpu_end.user_s - cpu_start.user_s, cpu_end.sys_s - cpu_start.sys_s };
    const int     cores        = cpu_core_count();
    const double  cpu_util_pct = cpu_used.total() / (wall_ms / 1e3) * 100.0 / cores;

    std::cout << "\n"
              << "Resource usage:\n"
              << std::fixed << std::setprecision(1)
              << "  Memory RSS       " << mem_start.rss_mb << " MB → " << mem_end.rss_mb << " MB"
              << "   peak " << mem_end.peak_rss_mb << " MB\n"
              << "  CPU utilisation  " << cpu_util_pct << "% of " << cores << " physical cores"
              << "  (" << cpu_used.user_s << " s user + " << cpu_used.sys_s << " s sys"
              << "  /  " << wall_ms / 1e3 << " s wall)\n";
    if (!gpu_util.empty()) {
        const auto gu = compute_stats(gpu_util);
        const auto gm = compute_stats(gpu_mem_mb);
        std::cout << "  GPU utilisation  mean " << gu.mean << "%   min " << gu.min << "%   max " << gu.max << "%\n"
                  << "  GPU mem (proc)   mean " << gm.mean << " MB  peak " << gm.max << " MB\n";
    } else {
        std::cout << "  GPU utilisation  N/A (NVML not available)\n";
    }
    std::cout << "=========================================\n";

    return 0;
}
