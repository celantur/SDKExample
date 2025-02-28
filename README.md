# SDKExample

## General
In this repository one can find a lot of example code that is used to use Celantur SDK. SDK consists of several shared objects and include files, that can be linked against and included in your program to use the functionality.

The easiest way to integrate the functionality is using the cmake. Add the `findPackage` directive to find the celantur library and link targets into your program, e.g.:
```bash
find_package(CppProcessing REQUIRED CelanturSDK common-module) 
...
target_link_libraries(YourExecutableOrLibrary PRIVATE CppProcessing::CelanturSDK CppProcessing::common-module)
```
For an example, use the `CMakeLists.txt` in this repository.

```bash
cmake -S /path/to/your/program/ -B /path/to/build/dir/ 
```

The dependencies are installed into the `/usr/local/` prefix. If the cmake `findPackage` returns an error, manually provide the path to the cmake file:
```bash
cmake -S /path/to/your/program/ -B /path/to/build/dir/ -DCppProcessing_DIR=/usr/local/lib/cmake
```

### Module overview
Normally, one needs two plugins to use SDK. First is `CppProcessing::CelanturSDK` that consists of the general interfaces to the SDK and used as entry point for interaction with it. Another one is `CppProcessing::common-module` and it consists of multiple definitions, classes and structures that are exposed to the user since they are common between our internal code and SDK.

Other shared objects do not provide the include files and are just passive dependency to the `CelanturSDK` shared object. Because of that, they are not really interesting to the end user.

### Plugin overview
Different modes of work require different dependencies installed on the target machine. For example, `libONNXInference.so` depends on [ONNX](https://onnx.ai/) to run the detection on the CPU. On the other hand, `libTensorRTRuntime.so` depends on Nvidia and CUDA libraries to provide detections on the GPU. To avoid having hard dependencies to these libraries, we use the plugin system where the dependencies are encapsulated in the plugin that is being loaded at runtime. You need to load at least one inference engine plugin for the SDK to work, which is quite straightforward and covered in tutorials.

