## OpenCV

If you are using a different version of OpenCV, distributed by e.g. package manager, we cannot guarantee that the SDK will work. If you rely on compiling OpenCV manually, you will need to ensure the following flags are set and the following modules are included in the `cmake` configuration:

```
# Dependency for DNN
-DBUILD_PROTOBUF:BOOL=ON 
# Dependency for videostab
-DBUILD_opencv_photo:BOOL=ON 
# Dependency for highgui
-DWITH_GTK_2_X:BOOL=ON 
# Dependency for videostab
-DBUILD_opencv_flann:BOOL=ON 
# Image hash
-DBUILD_opencv_img_hash:BOOL=ON 
# Dependency for videostab
-DBUILD_opencv_features2d:BOOL=ON 
#Dependency for videostab
-DBUILD_opencv_ccalib:BOOL=ON 
# Dependency for videostab
-DBUILD_opencv_calib3d:BOOL=ON 
# Just in case
-DWITH_GSTREAMER:BOOL=ON
# GTK is manually enabled for HighGUI module. You can disable it if you don't need it
-DWITH_GTK:BOOL=ON
-DWITH_GTK_2_X:BOOL=ON
# FFMpeg is not strictly required but it did not hurt to have it.
-DWITH_FFMEG:BOOL=ON
```

Note that highgui is not a strict dependency and you can remove it if you don't need it. Also, one of the modules `opencv_videostab` is not a part of core opencv and it comes from OpenCV_contrib repository. You can read more on how to compile OpenCV with contrib modules [at official repository](https://github.com/opencv/opencv_contrib).

## TensorRT
Technically, the `celantur-tensorrt` package is not needed and provided only for convenience. If you have TensorRT installed on your system, you can skip this step. Note that different versions of CelanturSDK might require different versions of TensorRT, so you need to communicate with that to Celantur to get the correct version. The minimal supported TensorRT version currently is `8.5`. 


### Version incompatibilites
Please note that the CUDA version on your machine is tied to the TensorRT version. NVidia supports only a limited number of CUDA versions for each TensorRT version. You can find the compatibility matrix [here](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html). If you have a different version of CUDA installed, you might need to install a different version of TensorRT.

#### x86_64
Celantur-go-to version of Tensort (that is provided with `celantur-tensorrt` package) is 10.0. Two different versions of TensorRT support CUDA `11` and `12` respectively. Celantur provided package expects you to have CUDA 11. If you have a different version of CUDA installed, ask Celantur for the correct version of TensorRT.
#### Jetson
Jetson has narrower support for TensorRT versions that are tied to your Jetpack version. The minimal requirement of `8.5` translates into JetPack version `5.1.2`. Newer versions of JetPack (starting from `6.0`) support TensorRT `10.0`. Each downgrade basically corresponds to the less features in the SDK.

** Important **: some features of the SDK might not work with the older versions of TensorRT versions. The machine learning is a fast-evolving field and we are constantly updating our models and SDK to provide the best possible performance.

** Important **: Older Jetsons are limited to JetPack `5`. Only Jetson Orin and newer support JetPack `6` and newer that can fully utilise CelanturSDK.