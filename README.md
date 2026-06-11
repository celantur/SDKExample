# Celantur SDK Examples

The [Celantur](https://www.celantur.com) SDK provides privacy-preserving image blurring through automatic anonymisation of faces and license plates.

This repository contains example programs that show how to integrate the [Celantur SDK](https://doc.celantur.com/sdk/getting-started) into your application.

## Prerequisites

The SDK is shipped as a Debian package. Shared libraries and headers are installed under `/usr/local/`.

See the [installation guide](https://doc.celantur.com/sdk/requirements-and-installation/installation) for setup instructions.

## Examples

Start with [quickstart.cpp](quickstart.cpp), then explore the other examples as needed.

| Example | What it demonstrates |
| --- | --- |
| [quickstart.cpp](quickstart.cpp) | Minimal CPU anonymisation with the ONNX inference engine, plus tuning inference settings such as thread count and optimisation level. |
| [jpeg.cpp](jpeg.cpp) | Full CPU workflow: JPEG decode/encode with EXIF preservation, detection visualisation, per-class counts, and metric serialisation. |
| [openvino.cpp](openvino.cpp) | Compile and run a model with the OpenVINO CPU inference engine. |
| [tensorrt.cpp](tensorrt.cpp) | Compile and run a model on GPU with TensorRT, including precision and optimisation level. |
| [tracking.cpp](tracking.cpp) | Video processing with object tracking using a smaller model. |
| [cuda.cpp](cuda.cpp) | Full GPU inference pipeline without copying image data back to the CPU. |

Shared paths, processor defaults, and the common OpenCV image-processing helper live in [example_common.h](example_common.h).

## Building

The easiest way to build the examples is with CMake:

```bash
mkdir -p build/
cmake -S . -B build/
cmake --build build/
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
| `v10-static-fp32-medium-1280.onnx.enc` | Encrypted model (default in the examples) |
| `image.jpg` | Test image |

If you were given custom models, update the model path in [example_common.h](example_common.h) or in the relevant example source file.

### Executables

After building, run the examples from `build/`:

```bash
./quickstart
./jpeg
./openvino
# ...
```

## Troubleshooting

### `CppProcessing` is not found

```
CMake Error at CMakeLists.txt:10 (find_package):
  By not providing "FindCppProcessing.cmake" in CMAKE_MODULE_PATH this
  project has asked CMake to find a package configuration file provided by
  "CppProcessing", but CMake did not find one.
```

**Solution:** point CMake at the SDK config directory:

```bash
cmake -S . -B build/ -DCppProcessing_DIR=/usr/local/lib/cmake
```

### Error loading shared libraries

```
error while loading shared libraries: libprocessing.so: cannot open shared object file
```

**Solution:** add the SDK library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Alternatively, adjust the executable [runpath](https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included).

## Additional settings

Logging defaults to `INFO`. To reduce verbosity:

```bash
export LOG_LEVEL=WARNING
```
