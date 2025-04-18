ONNX CPU Plugin
===============

## Install dependencies (on Ubuntu 22.04)

### Requirements
```bash
sudo apt install libprotobuf-dev libopenexr25 libgstreamer1.0-0 ffmpeg  libgstreamer-plugins-base1.0-0 libexif-dev
```

### Optional
```bash
sudo apt install cmake git build-essential ninja-build
```

### Celantur packages

```bash
apt install ./celantur-boost_1.84.0-x86_64.deb ./celantur-onnx_1.20.1.deb ./celantur-opencv_4.7.0-x86_64.deb
```

```bash
apt install ./celantur-cpp-processing-<your-distro>.deb
```

## Usage
1. Load the CPU processing plugin, by specifying `ProcessorParams.inference_plugin = /usr/local/lib/libONNXInference.so;`
2. Load the inference model (file ending with `.onnx.enc`) using `CelanturSDK::Processor.load_inference_model`.
3. After initialising the processor class, you can start anonymisuing images. 
   More settings and tinkering are available in different examples in this repository.
