Celantur SDK Examples
=====================

The [Celantur](https://www.celantur.com) SDK provides privacy-preserving image blurring through
automatic anonymisation of faces and license plates. 
(Person and vehicle anonymisation is currently possible with [Celantur Container](https://www.celantur.com/container/) and coming soon to the SDK.)

In this repository you find a lot of example code how to use [Celantur SDK](https://doc.celantur.com/sdk/getting-started).


## Prerequisite

The SDK consists of several shared objects and include files, 
that can be linked against and included in your program.
It is shipped as a Debian packaged and the files are extracted to `/usr/local/`. 

For CPU processing with ONNX models you find the instructions in [ONNXCPUPlugins.md](./ONNXCPUPlugin.md).

The easiest way to integrate the functionalities is using CMake. 

First, create CMake configuration files:

```bash
# Create configuration files
cmake -S /path/to/your/program/ -B /path/to/build/dir/ 
# Build the binaries
cd /path/to/build/dir/
make
```

### CMake

In `CMakeLists.txt` you can use the `findPackage` directive to add libraries and link targets 
into your program, e.g.:

```bash
find_package(CppProcessing REQUIRED CelanturSDK common-module) 
...
target_link_libraries(YourExecutableOrLibrary PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)
```

If CMake `findPackage` returns an error, check out the [Troubleshooting](#troubleshooting) section.


## Run examples

Copy a JPEG image named `image.jpg` to `/path/to/build/dir/` and run the script.

In `/path/to/build/dir/`, you find the following executables:
- `quickstart`: Blurs the image.
- `processor_parameters_detail`: Same as quickstart but with more parameters 
  (2x2 tiling and blurring of only the top-left quarter of the image).
- `decoding_encoding`: Image is encoded and decoded with the SDK (instead of OpenCV),
  and the metadata is conserved in the output image.
- `inference_tinkering`: The code shows how to configure the parameters of the inference engine.
- `detections_and_thresholds`: The code shows how to filter detections based on threshold,
  to visualise detections and to generate detection metrics in JSON.


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
cmake -GNinja -DCMAKE_INSTALL_PREFIX=path/to/install/ -DCppProcessing_DIR=/usr/local/lib/cmake ..
```

### Error loading shared libraries

```
./celantur_sdk_example: error while loading shared libraries: libprocessing.so: cannot open shared object file: No such file or directory`
```

#### Solution

Add the library path to the install celantur library:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Or edit the [runpath](https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included).




