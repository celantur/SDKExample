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
sudo apt-get update && sudo apt install -y ffmpeg python3-dev cmake ninja-build
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


