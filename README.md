# SDKExample

## First setup
This is a simple example of how to use the Celantur SDK. The SDK is a library that allows you to easily integrate Celantur's anonymisation services into your app. To start, you will need to receive several things from Celantur:
1. Model file
2. Custom-made OpenCV debian package `celantur-opencv.deb`. This package is a modified version of OpenCV 4.7.0 that includes the necessary functions for the SDK to work. 
3. Celantur SDK library for C++, `celantur-cpp-processing.deb`
4. Depending on your system you might also get a `celantur-tensorrt.deb` package. If you already have TensorRT installed on your system, you can skip this step.

Make sure that your system has CUDA installed. Celantur does not impose any specific version of CUDA, but there can be incompatibilities with different versions of TensorRT. Read more about this in the [Dependencies.md](Dependencies.md).

## Known dependency limitations
- You can read more about the dependencies in the [Dependencies.md](Dependencies.md).

## Installing dependencies
1. *Not necessary* if not needed remove the `python2` dependencies that might create conflicts with the SDK:
```bash
sudo apt remove python2* libpython2*
```
2. Remove preinstalled **OpenCV** since we will use a custom version (read more about it [Dependencies.md](Dependencies.md)):
```bash
sudo apt-get remove libopencv*
```

3. install the required repository dependencies:
```bash
sudo apt-get update && sudo apt install -y ffmpeg python3-dev cmake ninja-build libeigen3-dev libboost-all-dev
```

4. Install the custom OpenCV package:
```bash
sudo apt install ./celantur-opencv*.deb
```

5. Install the Celantur SDK:
```bash
sudo apt install ./celantur-cpp-processing*.deb
```

## Compile 
```bash
mkdir -p build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=path/to/install/ ..
ninja && ninja install
```

### Troubleshooting

#### CppProcessing is not found

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

## Compile model
Before running the example, one needs to compile the model. The model that Celantur provides is in universal, 
hardware independent format `.onnx`. However, to achieve the best performance we use the inference engine of 
NVidia called **TensorRT**. 

There are few parameters that may or may not be supported on all **TensorRT** versions:
1. If the version is higher than `8.6`, you can set *builder optimisation level*.
2. If the version is higher than `10`, dynamic model resolution will be supported.

To convert model from `.onnx` to `.trt` one need to compile it. Compiling the model in the context of Machine Learning
means finding the best possible deployment on a given hardware and as such, must be performed on a target hardware.
Model conversion needs to be done only once per hardware for any given set of parameters.

To compile the model, use the following cli command:
```bash
/usr/local/bin/create_trt_from_onnx path/to/onnx.onnx \
                                    path/to/tensorrt-output.trt \
                                    <precision> \
                                    <builder optimisation> \
                                    width=min:opt:max \
                                    height=min:opt:max
```

Before going over all parameters, we can provide some "safe" config which one can use as to quickly run the SDK example without setting it up too much:

```bash
/usr/local/bin/create_trt_from_onnx path/to/onnx.onnx \
                                    path/to/tensorrt-output.trt \
                                    FP32 \
                                    0 \
                                    width=1280:1280:1280 \
                                    height=1280:1280:1280
```

First parameter in the list, `<precision>` has a two options `FP16` and `FP32`. It sets the recommended precision and performance of the resulting model. `FP16` on average will be faster but less precise, `FP32` will be more precise but less performant.

Second parameter, `<builder optimisation>`, denotes the optimisation of the resulting model. Can be values from 0 to 5. The higher it is, the longer the model compilation will take, but the faster the inference phase will be. If you have **TensorRT** version < `8.6` this parameter will not influence the result, but still needs to be provided.

Last two parameters denote minimum/optimum/maximum model resolution of dynamic model. We suggest to start with fixed resolution `1280:1280:1280` and figure out if different resolution is needed later. The larger the resolution is, the bigger image the model can process without loss of precision.

Won't work if your TensorRT version is less than 10. If it is the case, please always use `1280:1280:1280` since in the earlier TensorRT versions a bug exists, that will make model with the same input to output some gibberish.

### Troubleshooting

#### Unable to load library

```
Using precision FP32 and optimisation level 0
Creating builder
6: [libLoader.cpp::Impl::291] Error Code 6: Internal Error (Unable to load library: libnvinfer_builder_resource.so.10.0.1: libnvinfer_builder_resource.so.10.0.1: cannot open shared object file: No such file or directory)
Segmentation fault (core dumped)
```

#### Solution

Add `/usr/local/lib` to `LD_LIBRARY_PATH`.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

## Run the SDK example
Finally, you can use compiled model to anonymise any image:
```bash
cd path/to/install
./celantur_sdk_example test-img.jpg path/to/tensorrt-output.trt
```

### Troubleshooting

#### Error loading shared libraries

```
./celantur_sdk_example: error while loading shared libraries: libprocessing.so: cannot open shared object file: No such file or directory`
```

#### Solution

Add the library path to the install celantur library:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Or edit the [runpath](https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included)




                                