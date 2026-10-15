#include "CelanturSDKInterface.h"
#include "CommonParameters.h"
#include "example_common.h"

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

/**
    Anonymise an HEVC video with the Celantur SDK (TensorRT) and FFmpeg for decode/encode.

    A single example covers both CPU and NVIDIA hardware codecs. Select them with:
      --decoder hevc|hevc_cuvid
      --encoder libx265|hevc_nvenc

    Default input/output are raw HEVC elementary streams (.h265), same form as the sample asset.
 */

namespace {

namespace po = boost::program_options;

[[noreturn]] void throw_av_error(const std::string& what, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, buf, sizeof(buf));
    throw std::runtime_error(what + ": " + buf);
}

void check_av(int err, const std::string& what) {
    if (err < 0) {
        throw_av_error(what, err);
    }
}

bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(value[value.size() - suffix.size() + i])));
        const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) {
            return false;
        }
    }
    return true;
}

AVPixelFormat pick_sw_pix_fmt(const AVCodec* codec) {
    if (codec->pix_fmts == nullptr) {
        return AV_PIX_FMT_YUV420P;
    }
    for (const AVPixelFormat* fmt = codec->pix_fmts; *fmt != AV_PIX_FMT_NONE; ++fmt) {
        if (*fmt == AV_PIX_FMT_YUV420P) {
            return AV_PIX_FMT_YUV420P;
        }
    }
    return codec->pix_fmts[0];
}

AVPixelFormat get_hw_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    const AVPixelFormat want = ctx->opaque != nullptr
        ? *static_cast<const AVPixelFormat*>(ctx->opaque)
        : AV_PIX_FMT_CUDA;
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == want) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

struct CliOptions {
    std::string input;
    std::string output;
    std::string decoder;
    std::string encoder;
};

std::string make_output_name(const std::string& input, const std::string& decoder, const std::string& encoder) {
    // e.g. assets/vid_h265.h265 + hevc + libx265 -> vid_h265_hevc_libx265.h265
    const std::string stem = std::filesystem::path(input).stem().string();
    return stem + "_" + decoder + "_" + encoder + ".h265";
}

CliOptions parse_cli(int argc, char** argv) {
    const std::string default_input = example::asset("vid_h265.h265").string();

    po::options_description desc("video_ffmpeg options");
    desc.add_options()
        ("help,h", "Show this help message")
        ("input,i", po::value<std::string>()->default_value(default_input), "Input video (raw .h265 or container)")
        ("decoder,d", po::value<std::string>()->default_value("hevc"), "Decoder: hevc (CPU) or hevc_cuvid (NVIDIA)")
        ("encoder,e", po::value<std::string>()->default_value("libx265"), "Encoder: libx265 (CPU) or hevc_nvenc (NVIDIA)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") != 0) {
        std::cout << "Anonymise HEVC video with Celantur SDK (TensorRT) and FFmpeg.\n\n"
                  << desc << "\n"
                  << "Output is written under output/ as {input_stem}_{decoder}_{encoder}.h265\n\n"
                  << "Examples:\n"
                  << "  ./video_ffmpeg\n"
                  << "  ./video_ffmpeg --decoder hevc --encoder libx265\n"
                  << "  ./video_ffmpeg --decoder hevc_cuvid --encoder hevc_nvenc\n";
        std::exit(0);
    }
    po::notify(vm);

    CliOptions opts{
        .input = vm["input"].as<std::string>(),
        .output = {},
        .decoder = vm["decoder"].as<std::string>(),
        .encoder = vm["encoder"].as<std::string>(),
    };

    if (opts.decoder != "hevc" && opts.decoder != "hevc_cuvid") {
        throw std::runtime_error("Unsupported --decoder '" + opts.decoder + "' (use hevc or hevc_cuvid)");
    }
    if (opts.encoder != "libx265" && opts.encoder != "hevc_nvenc") {
        throw std::runtime_error("Unsupported --encoder '" + opts.encoder + "' (use libx265 or hevc_nvenc)");
    }

    opts.output = example::output_file(make_output_name(opts.input, opts.decoder, opts.encoder)).string();
    return opts;
}

struct VideoIO {
    AVFormatContext* in_fmt{nullptr};
    AVFormatContext* out_fmt{nullptr};
    AVCodecContext* dec_ctx{nullptr};
    AVCodecContext* enc_ctx{nullptr};
    AVBufferRef* hw_device_ctx{nullptr};
    AVPixelFormat hw_pix_fmt{AV_PIX_FMT_NONE};
    SwsContext* to_bgr{nullptr};
    SwsContext* from_bgr{nullptr};
    AVFrame* dec_frame{nullptr};
    AVFrame* sw_frame{nullptr};
    AVFrame* enc_frame{nullptr};
    AVPacket* packet{nullptr};
    int in_video_index{-1};
    int out_video_index{-1};
    AVRational in_time_base{};
    AVRational frame_rate{.num = 25, .den = 1};
    int64_t next_pts{0};
    AVPixelFormat to_bgr_src_fmt{AV_PIX_FMT_NONE};
    int to_bgr_w{0};
    int to_bgr_h{0};

    ~VideoIO() {
        close();
    }

    void close() {
        if (to_bgr != nullptr) {
            sws_freeContext(to_bgr);
            to_bgr = nullptr;
        }
        if (from_bgr != nullptr) {
            sws_freeContext(from_bgr);
            from_bgr = nullptr;
        }
        av_frame_free(&dec_frame);
        av_frame_free(&sw_frame);
        av_frame_free(&enc_frame);
        av_packet_free(&packet);
        avcodec_free_context(&dec_ctx);
        avcodec_free_context(&enc_ctx);
        av_buffer_unref(&hw_device_ctx);
        if (out_fmt != nullptr) {
            if ((out_fmt->oformat->flags & AVFMT_NOFILE) == 0 && out_fmt->pb != nullptr) {
                avio_closep(&out_fmt->pb);
            }
            avformat_free_context(out_fmt);
            out_fmt = nullptr;
        }
        if (in_fmt != nullptr) {
            avformat_close_input(&in_fmt);
        }
    }

    void open_input(const std::string& path, const std::string& decoder_name) {
        // FFmpeg 4.x still takes a non-const AVInputFormat* here.
        AVInputFormat* forced_format = nullptr;
        if (ends_with_ci(path, ".h265") || ends_with_ci(path, ".hevc")) {
            forced_format = const_cast<AVInputFormat*>(av_find_input_format("hevc"));
        }

        check_av(avformat_open_input(&in_fmt, path.c_str(), forced_format, nullptr), "avformat_open_input");
        check_av(avformat_find_stream_info(in_fmt, nullptr), "avformat_find_stream_info");

        in_video_index = av_find_best_stream(in_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (in_video_index < 0) {
            throw std::runtime_error("No video stream in " + path);
        }

        AVStream* in_stream = in_fmt->streams[in_video_index];
        in_time_base = in_stream->time_base;
        const AVRational fr = (in_stream->avg_frame_rate.num > 0) ? in_stream->avg_frame_rate : in_stream->r_frame_rate;
        if (fr.num > 0 && fr.den > 0) {
            frame_rate = fr;
        }

        const AVCodec* decoder = avcodec_find_decoder_by_name(decoder_name.c_str());
        if (decoder == nullptr) {
            throw std::runtime_error("Decoder not found in this FFmpeg build: " + decoder_name);
        }

        dec_ctx = avcodec_alloc_context3(decoder);
        if (dec_ctx == nullptr) {
            throw std::runtime_error("avcodec_alloc_context3 failed for decoder");
        }
        check_av(avcodec_parameters_to_context(dec_ctx, in_stream->codecpar), "avcodec_parameters_to_context");
        dec_ctx->pkt_timebase = in_time_base;

        if (decoder_name.find("cuvid") != std::string::npos) {
            check_av(
                av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0),
                "av_hwdevice_ctx_create (CUDA)");
            dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
            if (dec_ctx->hw_device_ctx == nullptr) {
                throw std::runtime_error("av_buffer_ref failed for hw_device_ctx");
            }
            hw_pix_fmt = AV_PIX_FMT_CUDA;
            dec_ctx->opaque = &hw_pix_fmt;
            dec_ctx->get_format = get_hw_format;
            dec_ctx->extra_hw_frames = 16;
        }

        check_av(avcodec_open2(dec_ctx, decoder, nullptr), "avcodec_open2 (decoder)");

        dec_frame = av_frame_alloc();
        sw_frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (dec_frame == nullptr || sw_frame == nullptr || packet == nullptr) {
            throw std::runtime_error("Failed to allocate decode frames/packet");
        }
    }

    void open_output(const std::string& path, const std::string& encoder_name) {
        // Force the raw HEVC muxer so the result is an elementary .h265 blob (Annex B),
        // matching the input asset rather than an MP4/Matroska container.
        check_av(avformat_alloc_output_context2(&out_fmt, nullptr, "hevc", path.c_str()), "avformat_alloc_output_context2");
        if (out_fmt == nullptr) {
            throw std::runtime_error("avformat_alloc_output_context2 returned null");
        }

        const AVCodec* encoder = avcodec_find_encoder_by_name(encoder_name.c_str());
        if (encoder == nullptr) {
            throw std::runtime_error("Encoder not found in this FFmpeg build: " + encoder_name);
        }

        AVStream* out_stream = avformat_new_stream(out_fmt, nullptr);
        if (out_stream == nullptr) {
            throw std::runtime_error("avformat_new_stream failed");
        }
        out_video_index = out_stream->index;

        enc_ctx = avcodec_alloc_context3(encoder);
        if (enc_ctx == nullptr) {
            throw std::runtime_error("avcodec_alloc_context3 failed for encoder");
        }

        enc_ctx->width = dec_ctx->width;
        enc_ctx->height = dec_ctx->height;
        enc_ctx->sample_aspect_ratio = dec_ctx->sample_aspect_ratio;
        enc_ctx->framerate = frame_rate;
        enc_ctx->time_base = AVRational{.num = frame_rate.den, .den = frame_rate.num};
        enc_ctx->bit_rate = 4'000'000;
        enc_ctx->gop_size = 30;
        enc_ctx->max_b_frames = (encoder_name.find("nvenc") != std::string::npos) ? 0 : 2;
        enc_ctx->pix_fmt = pick_sw_pix_fmt(encoder);

        // Do not set AV_CODEC_FLAG_GLOBAL_HEADER: raw HEVC needs in-band VPS/SPS/PPS (Annex B).

        // Prefer a stable quality setting when the encoder exposes CRF/CQ.
        if (encoder_name == "libx265") {
            av_opt_set(enc_ctx->priv_data, "crf", "23", 0);
            av_opt_set(enc_ctx->priv_data, "preset", "medium", 0);
        } else if (encoder_name == "hevc_nvenc") {
            av_opt_set(enc_ctx->priv_data, "preset", "p4", 0);
            av_opt_set(enc_ctx->priv_data, "tune", "hq", 0);
            av_opt_set(enc_ctx->priv_data, "rc", "vbr", 0);
        }

        check_av(avcodec_open2(enc_ctx, encoder, nullptr), "avcodec_open2 (encoder)");
        check_av(avcodec_parameters_from_context(out_stream->codecpar, enc_ctx), "avcodec_parameters_from_context");
        out_stream->time_base = enc_ctx->time_base;

        if ((out_fmt->oformat->flags & AVFMT_NOFILE) == 0) {
            check_av(avio_open(&out_fmt->pb, path.c_str(), AVIO_FLAG_WRITE), "avio_open");
        }
        check_av(avformat_write_header(out_fmt, nullptr), "avformat_write_header");

        enc_frame = av_frame_alloc();
        if (enc_frame == nullptr) {
            throw std::runtime_error("av_frame_alloc failed for encoder frame");
        }
        enc_frame->format = enc_ctx->pix_fmt;
        enc_frame->width = enc_ctx->width;
        enc_frame->height = enc_ctx->height;
        check_av(av_frame_get_buffer(enc_frame, 0), "av_frame_get_buffer");

        from_bgr = sws_getContext(
            enc_ctx->width, enc_ctx->height, AV_PIX_FMT_BGR24,
            enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (from_bgr == nullptr) {
            throw std::runtime_error("sws_getContext failed (BGR -> encoder)");
        }
    }

    cv::Mat frame_to_bgr(const AVFrame* src) {
        const AVFrame* usable = src;
        AVFrame* downloaded = nullptr;

        if (src->format == AV_PIX_FMT_CUDA || src->format == AV_PIX_FMT_VAAPI || src->format == AV_PIX_FMT_QSV) {
            downloaded = sw_frame;
            av_frame_unref(downloaded);
            check_av(av_hwframe_transfer_data(downloaded, src, 0), "av_hwframe_transfer_data");
            usable = downloaded;
        }

        const auto src_fmt = static_cast<AVPixelFormat>(usable->format);
        if (to_bgr == nullptr || src_fmt != to_bgr_src_fmt || usable->width != to_bgr_w || usable->height != to_bgr_h) {
            sws_freeContext(to_bgr);
            to_bgr = sws_getContext(
                usable->width, usable->height, src_fmt,
                usable->width, usable->height, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (to_bgr == nullptr) {
                throw std::runtime_error("sws_getContext failed (decoder -> BGR)");
            }
            to_bgr_src_fmt = src_fmt;
            to_bgr_w = usable->width;
            to_bgr_h = usable->height;
        }

        cv::Mat bgr(usable->height, usable->width, CV_8UC3);
        uint8_t* dst_data[4] = {bgr.data, nullptr, nullptr, nullptr};
        int dst_linesize[4] = {static_cast<int>(bgr.step[0]), 0, 0, 0};
        sws_scale(to_bgr, usable->data, usable->linesize, 0, usable->height, dst_data, dst_linesize);
        return bgr;
    }

    void write_encoded_packets(AVPacket* pkt) {
        while (true) {
            const int err = avcodec_receive_packet(enc_ctx, pkt);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            check_av(err, "avcodec_receive_packet");

            pkt->stream_index = out_video_index;
            av_packet_rescale_ts(pkt, enc_ctx->time_base, out_fmt->streams[out_video_index]->time_base);
            check_av(av_interleaved_write_frame(out_fmt, pkt), "av_interleaved_write_frame");
            av_packet_unref(pkt);
        }
    }

    void encode_bgr(const cv::Mat& bgr) {
        if (bgr.type() != CV_8UC3 || bgr.cols != enc_ctx->width || bgr.rows != enc_ctx->height) {
            throw std::runtime_error("Unexpected frame size/type for encoder");
        }

        check_av(av_frame_make_writable(enc_frame), "av_frame_make_writable");
        const cv::Mat contiguous = bgr.isContinuous() ? bgr : bgr.clone();
        const uint8_t* src_data[4] = {contiguous.data, nullptr, nullptr, nullptr};
        const int src_linesize[4] = {static_cast<int>(contiguous.step[0]), 0, 0, 0};
        sws_scale(from_bgr, src_data, src_linesize, 0, contiguous.rows, enc_frame->data, enc_frame->linesize);

        enc_frame->pts = next_pts++;

        AVPacket* pkt = av_packet_alloc();
        if (pkt == nullptr) {
            throw std::runtime_error("av_packet_alloc failed");
        }
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> pkt_guard(pkt, [](AVPacket* p) {
            av_packet_free(&p);
        });

        while (true) {
            const int err = avcodec_send_frame(enc_ctx, enc_frame);
            if (err == AVERROR(EAGAIN)) {
                write_encoded_packets(pkt);
                continue;
            }
            check_av(err, "avcodec_send_frame");
            break;
        }
        write_encoded_packets(pkt);
    }

    void flush_encoder() {
        AVPacket* pkt = av_packet_alloc();
        if (pkt == nullptr) {
            throw std::runtime_error("av_packet_alloc failed");
        }
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> pkt_guard(pkt, [](AVPacket* p) {
            av_packet_free(&p);
        });

        check_av(avcodec_send_frame(enc_ctx, nullptr), "avcodec_send_frame flush");
        write_encoded_packets(pkt);
        check_av(av_write_trailer(out_fmt), "av_write_trailer");
    }
};

CelanturSDK::Processor make_tensorrt_processor() {
    const std::filesystem::path model_path_compiled = example::asset("v11-static-fp16-medium-1280.trt.enc");

    if (!std::filesystem::exists(model_path_compiled)) {
        CelanturSDK::ModelCompilerParams compiler_params;
        compiler_params.inference_plugin = example::tensorrt_plugin;
        CelanturSDK::ModelCompiler compiler(example::license_file, compiler_params);
        celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(example::onnx_file_path);

        // Fast-compiling profile for the example; switch to FP16 + Full for production throughput.
        settings["precision"] = celantur::CompilePrecision::FP16;
        settings["optimisation_level"] = celantur::OptimisationLevel::Full;

        std::cout << "Compiling TensorRT model to " << model_path_compiled << std::endl;
        compiler.compile_model(settings, model_path_compiled);
    }

    celantur::ProcessorParams params = example::make_processor_params(example::tensorrt_plugin);
    params.n_tiles_x = 1;
    params.n_tiles_y = 1;
    params.ol_x = 0;
    params.ol_y = 0;
    // OpenCV/FFmpeg path below yields BGR; SDK expects RGB when swapRB is true.
    params.swapRB = true;

    std::cout << "Looking for license at " << example::license_file << std::endl;
    CelanturSDK::Processor processor(params, example::license_file);

    celantur::InferenceEnginePluginSettings settings = processor.get_inference_settings(model_path_compiled);
    example::print_inference_settings(settings);

    std::cout << "load model from " << model_path_compiled << std::endl;
    processor.load_inference_model(settings);
    return processor;
}

void process_decoded_frame(VideoIO& io, CelanturSDK::Processor& processor, AVFrame* frame, int& counter) {
    cv::Mat bgr = io.frame_to_bgr(frame);
    std::cout << "Processing frame " << ++counter << std::endl;

    processor.process(bgr);
    cv::Mat anonymised = processor.get_result();
    processor.get_detections(); // discard; frees detection memory

    io.encode_bgr(anonymised);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions opts = parse_cli(argc, argv);

        std::cout << "Input:    " << opts.input << "\n"
                  << "Output:   " << opts.output << "\n"
                  << "Decoder:  " << opts.decoder << "\n"
                  << "Encoder:  " << opts.encoder << "\n"
                  << "Inference: TensorRT\n";

        CelanturSDK::Processor processor = make_tensorrt_processor();

        VideoIO io;
        io.open_input(opts.input, opts.decoder);
        std::filesystem::create_directories(std::filesystem::path(opts.output).parent_path());
        io.open_output(opts.output, opts.encoder);

        std::cout << "Video: " << io.dec_ctx->width << "x" << io.dec_ctx->height
                  << " @ " << io.frame_rate.num << "/" << io.frame_rate.den << " fps\n";

        const auto t0 = std::chrono::steady_clock::now();
        int counter = 0;

        while (av_read_frame(io.in_fmt, io.packet) >= 0) {
            std::unique_ptr<AVPacket, void(*)(AVPacket*)> unref(io.packet, [](AVPacket* p) {
                av_packet_unref(p);
            });
            if (io.packet->stream_index != io.in_video_index) {
                continue;
            }

            while (true) {
                const int send_err = avcodec_send_packet(io.dec_ctx, io.packet);
                if (send_err == AVERROR(EAGAIN)) {
                    while (true) {
                        const int recv_err = avcodec_receive_frame(io.dec_ctx, io.dec_frame);
                        if (recv_err == AVERROR(EAGAIN) || recv_err == AVERROR_EOF) {
                            break;
                        }
                        check_av(recv_err, "avcodec_receive_frame");
                        process_decoded_frame(io, processor, io.dec_frame, counter);
                        av_frame_unref(io.dec_frame);
                    }
                    continue;
                }
                check_av(send_err, "avcodec_send_packet");
                break;
            }

            while (true) {
                const int recv_err = avcodec_receive_frame(io.dec_ctx, io.dec_frame);
                if (recv_err == AVERROR(EAGAIN) || recv_err == AVERROR_EOF) {
                    break;
                }
                check_av(recv_err, "avcodec_receive_frame");
                process_decoded_frame(io, processor, io.dec_frame, counter);
                av_frame_unref(io.dec_frame);
            }
        }

        // Flush decoder
        check_av(avcodec_send_packet(io.dec_ctx, nullptr), "avcodec_send_packet flush");
        while (true) {
            const int recv_err = avcodec_receive_frame(io.dec_ctx, io.dec_frame);
            if (recv_err == AVERROR(EAGAIN) || recv_err == AVERROR_EOF) {
                break;
            }
            check_av(recv_err, "avcodec_receive_frame flush");
            process_decoded_frame(io, processor, io.dec_frame, counter);
            av_frame_unref(io.dec_frame);
        }

        io.flush_encoder();

        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double fps = (ms > 0.0 && counter > 0) ? (1000.0 * static_cast<double>(counter) / ms) : 0.0;
        std::cout << "Done. frames=" << counter << "  wall=" << ms << " ms  throughput=" << fps << " fps\n"
                  << "Wrote " << opts.output << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
