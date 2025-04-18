TensorRT Inference Plugin
=========================

WARNING: COMING SOON! Currently not available in the SDK.

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