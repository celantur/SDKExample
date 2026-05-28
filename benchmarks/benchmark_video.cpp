#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ============================================================
// USER CONFIGURATION
// Edit create_processor() (and create_tracking() if using --tracking).
// Do not modify main() below.
// ============================================================

std::unique_ptr<CelanturSDK::Processor> create_processor() {
    const std::filesystem::path assets_path = std::filesystem::path(boost::dll::program_location().parent_path().string()) / ".." / ".." / "assets";
    const std::filesystem::path license  = assets_path / "license";
    const std::filesystem::path model    = assets_path / "v6-static-fp32-medium-1280.onnx.enc";
    const std::filesystem::path compiled = assets_path / "v6-static-fp32-medium-1280.openvino";
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

// Only called when --tracking is passed.
// Use the same license and per_type_processing_config as create_processor().
struct TrackingSetup {
    std::unique_ptr<CelanturSDK::Tracker>    tracker;
    std::unique_ptr<CelanturSDK::Anonymiser> anonymiser;
};

TrackingSetup create_tracking() {
    const std::filesystem::path license = std::filesystem::path(boost::dll::program_location().parent_path().string()) / ".." / ".." / "assets" / "license";

    celantur::ProcessorParams params; // TODO: apply same config as create_processor() if needed
    return {
        std::make_unique<CelanturSDK::Tracker>(license),
        std::make_unique<CelanturSDK::Anonymiser>(license, params.per_type_processing_config)
    };
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
        ("help,h",    "Show this help message")
        ("input,i",   po::value<std::string>()->required(),        "Video file to process")
        ("output,o",  po::value<std::string>()->default_value(""), "Path for anonymised output video (optional)")
        ("tracking",  po::bool_switch()->default_value(false),     "Enable tracking + re-anonymisation of lost bounding boxes");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << "benchmark_video: measure end-to-end video anonymisation performance\n\n"
                      << desc << "\n"
                      << "Edit create_processor() in the source file to configure the SDK.\n"
                      << "Edit create_tracking() as well when using --tracking.\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    const std::string input_str    = vm["input"].as<std::string>();
    const std::string output_str   = vm["output"].as<std::string>();
    const bool        use_tracking = vm["tracking"].as<bool>();

    if (!std::filesystem::exists(input_str)) {
        std::cerr << "Error: file not found: " << input_str << "\n";
        return 1;
    }

    std::cout << "Setting up processor...\n";
    auto processor = create_processor();

    std::unique_ptr<CelanturSDK::Tracker>    tracker;
    std::unique_ptr<CelanturSDK::Anonymiser> anonymiser;
    if (use_tracking) {
        std::cout << "Setting up tracker and anonymiser...\n";
        auto ts    = create_tracking();
        tracker    = std::move(ts.tracker);
        anonymiser = std::move(ts.anonymiser);
    }

    cv::VideoCapture cap(input_str);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video: " << input_str << "\n";
        return 1;
    }

    const int    frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double source_fps   = cap.get(cv::CAP_PROP_FPS);
    const int    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << frame_width << "x" << frame_height
              << "  source FPS: " << source_fps
              << "  frames: "     << total_frames
              << (use_tracking ? "  [tracking enabled]" : "") << "\n";

    std::unique_ptr<cv::VideoWriter> writer;
    if (!output_str.empty()) {
        std::filesystem::create_directories(
            std::filesystem::path(output_str).parent_path());
        writer = std::make_unique<cv::VideoWriter>(
            output_str,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            source_fps,
            cv::Size(frame_width, frame_height));
        if (!writer->isOpened()) {
            std::cerr << "Warning: could not open output writer for " << output_str << "; skipping output.\n";
            writer.reset();
        }
    }

    std::cout << "Starting benchmark...\n\n";

    std::vector<double> decode_times, process_times, tracking_times, save_times;
    cv::Mat frame;
    int frame_idx = 0;

    const auto wall_start = Clock::now();

    while (true) {
        const auto t0     = Clock::now();
        const bool got    = cap.read(frame);
        const auto t1     = Clock::now();

        if (!got || frame.empty()) break;

        processor->process(frame);
        cv::Mat out = processor->get_result();
        std::vector<celantur::CelanturDetection> dets = processor->get_detections();
        const auto t2 = Clock::now();

        double tracking_ms = 0.0;
        if (use_tracking) {
            tracker->update(dets);
            for (const auto& det : tracker->get_lost_detections())
                anonymiser->anonymise(out, det);
            tracking_ms = elapsed_ms(t2, Clock::now());
            tracking_times.push_back(tracking_ms);
        }

        decode_times.push_back(elapsed_ms(t0, t1));
        process_times.push_back(elapsed_ms(t1, t2));

        double save_ms = 0.0;
        if (writer) {
            const auto t_save = Clock::now();
            writer->write(out);
            save_ms = elapsed_ms(t_save, Clock::now());
            save_times.push_back(save_ms);
        }

        ++frame_idx;

        if (frame_idx == 1 || frame_idx % 50 == 0) {
            std::cout << "  frame " << std::setw(6) << frame_idx
                      << "  decode: "  << std::fixed << std::setprecision(1)
                      << std::setw(7) << decode_times.back()  << " ms"
                      << "  process: " << std::setw(7) << process_times.back() << " ms";
            if (use_tracking)
                std::cout << "  tracking: " << std::setw(7) << tracking_ms << " ms";
            if (writer)
                std::cout << "  save: " << std::setw(7) << save_ms << " ms";
            std::cout << "\n";
        }
    }

    const double wall_ms = elapsed_ms(wall_start, Clock::now());

    if (process_times.empty()) {
        std::cerr << "No frames were processed.\n";
        return 1;
    }

    const auto proc_s   = compute_stats(process_times);
    const auto decode_s = compute_stats(decode_times);

    const double inference_fps   = 1000.0 / proc_s.mean;
    const double end_to_end_fps  = 1000.0 / (proc_s.mean + decode_s.mean);

    std::cout << "\n"
              << "=========================================\n"
              << "           BENCHMARK RESULTS\n"
              << "=========================================\n"
              << "Input            : " << input_str << "\n"
              << "Resolution       : " << frame_width << "x" << frame_height << "\n"
              << std::fixed << std::setprecision(2)
              << "Source FPS       : " << source_fps << "\n"
              << "Frames processed : " << frame_idx << "\n"
              << "Wall-clock time  : " << wall_ms      / 1000.0 << " s\n"
              << "Total proc time  : " << proc_s.total / 1000.0 << " s\n"
              << "\n"
              << "Per-frame timing:\n";
    print_stats_row("decode",  decode_s);
    print_stats_row("process", proc_s);
    if (use_tracking && !tracking_times.empty())
        print_stats_row("tracking", compute_stats(tracking_times));
    if (!save_times.empty())
        print_stats_row("save", compute_stats(save_times));
    std::cout << "\n"
              << "Throughput:\n"
              << "  " << std::setw(6) << inference_fps  << " FPS  (process time only)\n"
              << "  " << std::setw(6) << end_to_end_fps << " FPS  (decode + process)\n"
              << "  " << std::setw(6) << source_fps     << " FPS  (source)\n"
              << "=========================================\n";

    return 0;
}
