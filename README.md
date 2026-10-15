# Celantur SDK Examples

The [Celantur](https://www.celantur.com) SDK provides privacy-preserving image blurring through automatic anonymisation of faces and license plates.

This repository contains example programs that show how to integrate the [Celantur SDK](https://doc.celantur.com/sdk/getting-started) into your application.

## Prerequisites

The SDK is shipped as a Debian package. Shared libraries and headers are installed under `/usr/local/`.

See the [installation guide](https://doc.celantur.com/sdk/requirements-and-installation/installation) for setup instructions.

## Examples

Start with [onnx.cpp](onnx.cpp), then explore the other examples as needed.

| Example | What it demonstrates |
| --- | --- |
| [onnx.cpp](onnx.cpp) | Minimal CPU anonymisation with the ONNX inference engine, plus tuning inference settings such as thread count and optimisation level. |
| [openvino.cpp](openvino.cpp) | Compile and run a model with the OpenVINO CPU inference engine. |
| [tensorrt.cpp](tensorrt.cpp) | Compile and run a model on GPU with TensorRT, including precision and optimisation level. |
| [cuda.cpp](cuda.cpp) | Full GPU inference pipeline without copying image data back to the CPU. |
| [tracking.cpp](tracking.cpp) | Video processing with object tracking using a smaller model. |
| [video_ffmpeg.cpp](video_ffmpeg.cpp) | HEVC video anonymisation with FFmpeg decode/encode (`hevc`/`hevc_cuvid` + `libx265`/`hevc_nvenc`) and TensorRT inference. |
| [jpeg.cpp](jpeg.cpp) | Full CPU workflow: JPEG decode/encode with EXIF preservation, detection visualisation, per-class counts, and metric serialisation. |

Shared paths, processor defaults, and the common OpenCV image-processing helper live in [`example_common.h`](example_common.h).

## Building

The easiest way to build the examples is with CMake:

```bash
mkdir -p build/
cmake -S . -B build/
cmake --build build/
```

If you receive the error
> By not providing "FindCppProcessing.cmake" in CMAKE_MODULE_PATH this
> project has asked CMake to find a package configuration file provided by
> "CppProcessing", but CMake did not find one.

... run:

```bash
cmake -S . -B build/ -DCppProcessing_DIR=/usr/local/lib/cmake
```

To link the SDK in your own project:

```cmake
find_package(CppProcessing REQUIRED CelanturSDK common-module)
target_link_libraries(YourExecutable PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)
```

If `find_package` fails, see [Troubleshooting](#troubleshooting).

## Running

### Assets

Copy the required files into `assets/`:

| File | Purpose |
| --- | --- |
| `license` | License file |
| `v11-static-fp32-medium-1280.onnx.enc` | Encrypted model (default in the examples) |
| `image.jpg` | Test image |
| `vid_h265.h265` | Raw HEVC test video (for `video_ffmpeg`) |
| `video.mp4` | Test video for `tracking.cpp` |

If you were given custom models, update the model path in [`example_common.h`](example_common.h) or in the relevant example source file.

### `video_ffmpeg` extras

`video_ffmpeg` needs FFmpeg development libraries (`libavcodec`, `libavformat`, `libavutil`, `libswscale`) at build time, and a runtime FFmpeg build that provides the codecs you select. Output is a raw `.h265` blob named `{input_stem}_{decoder}_{encoder}.h265` under `output/`.

```bash
# CPU path -> output/vid_h265_hevc_libx265.h265
./video_ffmpeg --decoder hevc --encoder libx265

# NVIDIA hardware path -> output/vid_h265_hevc_cuvid_hevc_nvenc.h265
./video_ffmpeg --decoder hevc_cuvid --encoder hevc_nvenc
```

### `video_opencv`

Uses OpenCV `VideoCapture` / `VideoWriter` (FFmpeg backend) for H.265→H.265. No codec CLI flags — OpenCV picks the libraries; optional:

```bash
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuda"
export OPENCV_FFMPEG_WRITER_OPTIONS="preset;medium|crf;23"
./video_opencv   # -> output/vid_h265_opencv.h265
```

Prints decode / SDK process / encode wall times and FPS.

### Executables

After building, run the examples from `build/`:

```bash
./onnx
./jpeg
./openvino
# ...
```

If you get
> error while loading shared libraries: libprocessing.so: cannot open shared object file

... add the SDK library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Alternatively, adjust the executable [runpath](https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included).


## Additional settings

Logging defaults to `INFO`. To reduce verbosity:

```bash
export LOG_LEVEL=WARNING
```
