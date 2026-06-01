Celantur SDK Examples
=====================

The [Celantur](https://www.celantur.com) SDK provides privacy-preserving image blurring through
automatic anonymisation of faces and license plates.

In this repository you find a lot of example code how to use [Celantur SDK](https://doc.celantur.com/sdk/getting-started).

## Prerequisite

The SDK consists of several shared objects and include files,
that can be linked against and included in your program.
It is shipped as a Debian package and the files are extracted to `/usr/local/`.

See [our documentation](https://doc.celantur.com/sdk/requirements-and-installation/installation)
for installation instructions.


## Examples

- `quickstart.cpp`: Use ONNX model and inference engine for a simple demonstration of object segmentation and blurring.
- `gpu_quickstart.cpp`: Compile ONNX to TensorRT model and run inference on GPU.
- `tiling.cpp`: Set up tiling and region-of-interest for inference.
- `object_types.cpp`: Configure object types and thresholds for inference and blurring.
- `onnx_compilation.cpp`: Configure compilation parameters for ONNX.
- `openvino_compilation.cpp`: Configure OpenVINO compilation.
- `openvino_small_model.cpp`: Configure OpenVINO compilation and inference for small models.
- `tensorrt_compilation.cpp`: Configure TensorRT compilation.
- `tensorrt_small_model.cpp`: Configure TensorRT compilation and inference for small models.
- `cuda.cpp`: Runs full inference on GPU without accessing CPU memory.
- `tracking.cpp`: Video processing and object tracking.
- `jpeg.cpp`: Proper JPEG processing with EXIF metadata.


## Building Examples

The easiest way to compile the examples is using CMake.

First, create CMake configuration files in the repo folder:

```bash
mkdir -p build/
# Create configuration files
cmake -S . -B build/
# Build the binaries
cd build/
make
```

In `CMakeLists.txt` you can use the `findPackage` directive to add libraries and link targets
into your program, e.g.:

```bash
find_package(CppProcessing REQUIRED CelanturSDK common-module)
...
target_link_libraries(YourExecutableOrLibrary PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)
```

If CMake `findPackage` returns an error, check out the [Troubleshooting](#troubleshooting) section.


## Run Examples

### Assets
Copy all assets required to `assets/`:
- `license`: License file.
- `v6-static-fp32.onnx.enc`: Encrypted model.
- `image.jpg`: Test images.

In case you were provided with some custom models, such as less-precise but faster models, your model name can be different.

You can change the file names in the source code.

### Executables

In `build/`, you find the examples.


## Troubleshooting

### CppProcessing is not found

```
CMake Error at CMakeLists.txt:10 (find_package):
  By not providing "FindCppProcessing.cmake" in CMAKE_MODULE_PATH this
  project has asked CMake to find a package configuration file provided by
  "CppProcessing", but CMake did not find one.

  Could not find a package configuration file provided by "CppProcessing"
  with any of the following names:

    CppProcessingConfig.cmake
    cppprocessing-config.cmake

  Add the installation prefix of "CppProcessing" to CMAKE_PREFIX_PATH or set
  "CppProcessing_DIR" to a directory containing one of the above files.  If
  "CppProcessing" provides a separate development package or SDK, be sure it
  has been installed.

```

#### Solution

Add `-DCppProcessing_DIR=/usr/local/lib/cmake` to the cmake configuration.

```bash
cmake -S . -B build/ -DCppProcessing_DIR=/usr/local/lib/cmake
```

### Error loading shared libraries

```
./celantur_sdk_example: error while loading shared libraries: libprocessing.so: cannot open shared object file: No such file or directory`
```

#### Solution

Add the library path to the linker paths:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Or edit the [runpath](https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included).

## Additional settings

By default, logging level is set to `INFO`. If you find it too verbose, you can change it to `WARNING` or `ERROR`. Here is an example how to set logging level to `WARNING`:

```bash
export LOG_LEVEL=WARNING
```
