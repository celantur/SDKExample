Celantur SDK Examples
=====================

The [Celantur](https://www.celantur.com) SDK provides privacy-preserving image blurring through
automatic anonymisation of faces and license plates.

> ! Person and vehicle anonymisation is currently (v1.1.0) buggy in combination with tiling.

In this repository you find a lot of example code how to use [Celantur SDK](https://doc.celantur.com/sdk/getting-started).

## Prerequisite

The SDK consists of several shared objects and include files,
that can be linked against and included in your program.
It is shipped as a Debian package and the files are extracted to `/usr/local/`.

See [Celantur Doc](https://doc.celantur.com/sdk/requirements-and-installation/installation)
for installation instructions.

## Building examples

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


## Run examples

### Assets
Copy all assets required to `assets/`:
- `license`: License file.
- `v6-static-fp32.onnx.enc`: Encrypted model.
- `image.jpg`: Test images.

You can change the file names in the source code.

### Executables

In `build/`, you find the following executables:
- `quickstart`: Blurs the image.
- `processor_parameters_detail`: Same as quickstart but with more parameters
  (2x2 tiling and blurring of only the top-left quarter of the image).
- `decoding_encoding`: Image is encoded and decoded with the SDK (instead of OpenCV),
  and the metadata is conserved in the output image.
- `inference_tinkering_onnx`: The code shows how to configure the parameters of the ONNX inference engine.
- `inference_tinkering_openvino`: The code shows how to configure the parameters of the OpenVINO inference engine.
- `detections_and_thresholds`: The code shows how to filter detections based on threshold,
  to visualise detections and to generate detection metrics in JSON.
- `person-vehicle-anon.cpp`: The code shows how to anonymise persons and vehicles with segmentation mask.

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




