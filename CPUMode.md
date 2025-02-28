##CPU Mode 

To use the CPU mode one requires to:
1. Install [CPU Dependencies](DependenciesCPU.md)
2. Load the CPU processing plugin. This is done by specifying the `ProcessorParams.inference_plugin = /usr/local/lib/libONNXInference.so;`
3. Load the inference model. For CPU processing the model is the one that ends with `.onnx.enc`. Use `CelanturSDK::Processor.load_inference_model`  to achieve that.
4. After initialising the processor class, you can start anonymisuing images. More settings and tinkering to get the results are available in different examples in this repository.
