#include "CelanturDetection.h"
#include "CelanturSDKInterface.h"
#include "CelanturSDKGPUInterface.h"
#include "CommonParameters.h"
#include "detections-utils.h"
#include <filesystem>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <boost/dll.hpp>

const std::filesystem::path exe_path = boost::dll::program_location().parent_path().string();
const std::filesystem::path assets_path = exe_path/".."/"assets";
const std::filesystem::path output_path = exe_path/".."/"output";
const std::filesystem::path gpu_plugin_location = "/app/output/lib/libTensorRTRuntime.so";
const std::filesystem::path license_file = assets_path/"license";
const std::filesystem::path image_path = assets_path/"image.jpg";
const std::filesystem::path out_image_path = output_path/"gpu_inference.jpg";
const std::filesystem::path model_path = assets_path/"v6-static-fp32.onnx.enc";
const std::filesystem::path model_path_compiled = assets_path/"v6-static-fp32-compiled-gpu.trt";

/**
 * This is just utility struct for the purpose of just copying the results from the GPU to the host and postprocessing them there. It is not necessary to do it this way, but it makes it easier to work with the results on the host side.
 * In a real gpu application you probably don't want this data to ever leave the GPU, but for the purpose of this example and to visualise things more easily, we copy the results to the host and postprocess them there.
 */
struct GPUInferenceResultHost2 : public celantur::GPUInferenceResult {
    std::vector<float> masks_;
    std::vector<float> scores_;
    std::vector<int32_t> classes_;
    std::vector<float> boxes_;
    
    explicit GPUInferenceResultHost2() = default; 
    GPUInferenceResultHost2(const GPUInferenceResult& other) : GPUInferenceResult(other) {}
    GPUInferenceResultHost2(GPUInferenceResultHost2&& other) = default;
    GPUInferenceResultHost2& operator=(const GPUInferenceResultHost2& other) = default;
    GPUInferenceResultHost2& operator=(GPUInferenceResultHost2&& other) = default;
    
    static GPUInferenceResultHost2 from_device_result(const celantur::GPUInferenceResultDevice& dev_res, cudaStream_t stream) {
        GPUInferenceResultHost2 host_res(
            static_cast<const GPUInferenceResult&>(dev_res)
        );
        
        ((GPUInferenceResult&) dev_res).print_dims(); // TODO: Remove this
        
        host_res.boxes_.resize(host_res.dims_.boxes_.totalSize());
        host_res.masks_.resize(host_res.dims_.masks_.totalSize());
        host_res.scores_.resize(host_res.dims_.scores_.totalSize());
        host_res.classes_.resize(host_res.dims_.classes_.totalSize());
        
        cudaMemcpyAsync(
            host_res.boxes_.data(), dev_res.boxes_,
            host_res.boxes_.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream
        );
        
        cudaMemcpyAsync(
            host_res.masks_.data(), dev_res.masks_,
            host_res.masks_.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream
        );
        
        
        cudaMemcpyAsync(
            host_res.scores_.data(), dev_res.scores_,
            host_res.scores_.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream
        );
        
        cudaMemcpyAsync(
            host_res.classes_.data(), dev_res.classes_,
            host_res.classes_.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream
        );
        
        return host_res;
    }
};

/**
 * This is also an utility function for the purpose of converting the raw outputs to the CelanturDetection format. We leave it here as an example of what you would need to extract detections, bounding boxes, masks, classes and so on.
*/
std::vector<celantur::CelanturDetection> postprocess_injected2(
    GPUInferenceResultHost2& host_res,
    cv::Size2i tile_size, 
    cv::Size2i context_size) 
{
    std::vector<celantur::CelanturDetection> detections;
    // num_detections is also present in the gpu result struct.
    if (host_res.num_detections == 0) return detections;
    detections.reserve(host_res.num_detections);
    
    // for scaling the bounding boxes to the original image size.
    int max_x = tile_size.width;
    int max_y = tile_size.height;
    float x_factor = static_cast<float>(tile_size.width) / static_cast<float>(context_size.width);
    float y_factor = static_cast<float>(tile_size.height) / static_cast<float>(context_size.height);
    
    
    // mask size is always the context size divided by 4. The raw inference output is always downscaled.
    const int mask_size = 320;
    
    for (int i = 0; i < host_res.num_detections; ++i) {
        const float* box_ptr = host_res.boxes_.data() + (i * 4);
        float x = box_ptr[0];
        float y = box_ptr[1];
        float w = box_ptr[2];
        float h = box_ptr[3];
        
        // Scale to image coordinates
        int left = static_cast<int>(x * x_factor);
        int top = static_cast<int>(y * y_factor);
        int width = static_cast<int>(w * x_factor);
        int height = static_cast<int>(h * y_factor);
        
        // Ensure bounding box stays within image boundaries (Clamping)
        cv::Rect roi_adj(left, top, width, height);
        if (roi_adj.x < 0) roi_adj.x = 0;
        if (roi_adj.y < 0) roi_adj.y = 0;
        if (roi_adj.x + roi_adj.width > max_x) roi_adj.width = max_x - roi_adj.x;
        if (roi_adj.y + roi_adj.height > max_y) roi_adj.height = max_y - roi_adj.y;
        
        // Extract and process the mask
        const float* mask_ptr = host_res.masks_.data() + (i * mask_size * mask_size);
        cv::Mat mask_mat(mask_size, mask_size, CV_32FC1);
        float* mask_data = reinterpret_cast<float*>(mask_mat.data);
        
        // The GPU output is already sigmoided (values are in [0.0, 1.0]).
        // We just do a strict binary threshold here.
        for (int j = 0; j < mask_size * mask_size; ++j) {
            mask_data[j] = (mask_ptr[j] > 0.5f) ? 1.0f : 0.0f;
        }
        
        // Resize the mask to the full tile size
        cv::Mat resized_mask;
        cv::resize(mask_mat, resized_mask, cv::Size(max_x, max_y), 0, 0, cv::INTER_NEAREST);
        
        // Crop the mask to the clamped bounding box. By default masks from the GPU are just masks for the whole image, but they are only valid within the bounding box.
        cv::Mat roi_mask = resized_mask(roi_adj).clone();
        
        // map to Celantur class ID and name
        celantur::YOLO_CLASS_IDs yolo_type = static_cast<celantur::YOLO_CLASS_IDs>(host_res.classes_[i]);
        celantur::CelanturClassId type_id = celantur::YOLO_CLASS_MAP.at(yolo_type);
        
        celantur::CelanturDetection result;
        result.class_id = static_cast<int>(type_id);
        result.class_name = celantur::CLASS_NAMES.at(type_id);
        result.model_class_id = host_res.classes_[i];
        result.box = roi_adj;
        result.segm_mask = roi_mask;
        result.confidence = host_res.scores_[i];
        
        detections.push_back(result);
    }
    
    return detections;
}

int main(int argc, char** argv) {
    // First, compile the model. This model is different from the one used in the CPU examples, as it is optimised for GPU inference. To check that you compiled correctly, check the output for:
    // [2026-04-20 06:04:43.248839] [0x00007701a0745000] [info]    Injecting Native NMS and Segmentation Graph...
    // [2026-04-20 06:04:43.249090] [0x00007701a0745000] [info]    NMS and Segmentation Graph Injection Complete.
    CelanturSDK::ModelCompilerParams params;
    params.inference_plugin = gpu_plugin_location;
    params.inference_plugin_group = celantur::PluginGroup::GPUInferenceEngine;
    
    CelanturSDK::ModelCompiler compiler(license_file, params);
    celantur::InferenceEnginePluginCompileSettings settings = compiler.preload_model(model_path);
    std::cout << "Compile settings: " << settings << std::endl;
    
    settings["precision"] = celantur::CompilePrecision::FP32;
    settings["optimisation_level"] = celantur::OptimisationLevel::Low;
    compiler.compile_model(settings, model_path_compiled);
    
    // Second, create the inference engine and load the compiled model.
    CelanturSDK::CUDAInferenceEngineParams cuda_params;
    cuda_params.inference_plugin = gpu_plugin_location;
    CelanturSDK::CUDAInferenceEngine engine(cuda_params, license_file);
    celantur::InferenceEnginePluginSettings rt_settings = engine.get_inference_settings(model_path_compiled);
    std::cout << "Runtime settings: " << rt_settings << std::endl;
    engine.load_inference_model(rt_settings);
    
    
    // Create a stream that we will use for all the GPU operations in this example. You can create multiple streams if you want to run multiple inferences in parallel, the workflow is the same.
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return -1;
    }
    CelanturSDK::AdditionalCUDAInferenceEngineParams additional_params;
    additional_params.context_height = 1280;
    additional_params.context_width = 1280;
    // This is an important step, because it tells the engine to prepare the execution context for the given stream and parameters. If you don't call this, the engine will throw an error. Note that execution context setup is a costly operation and also takes non-negligible amount of GPU memory. We take care that there is only one execution context prepared for a given stream, but using multiple streams might be a cause for OOM.
    engine.prepare_execution_context(stream, additional_params);
    std::cout << "Prepared execution context for stream " << stream << std::endl;
    
    cv::Mat img = cv::imread(image_path);
    
    uint8_t* d_img = nullptr;
    size_t img_bytes = img.step * img.rows;
    if (cudaMallocAsync(&d_img, img_bytes, stream) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for image" << std::endl;
        return -1;
    }
    
    if (cudaMemcpyAsync(d_img, img.data, img_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "Failed to copy image data to device" << std::endl;
        return -1;
    }
    
    // Now we need to preprocess the image into the tile that is a size of the context (1280x1280 in this case). The preprocess function runs on the GPU and takes care of resizing the image to the context size, normalising it and converting it to RGB if necessary. The output of the preprocess function is a tensor that is ready to be fed into the inference engine. 
    float* d_input = nullptr;
    celantur::Dims in_dims = engine.get_input_dimensions(stream);
    size_t input_bytes = in_dims.totalSize() * sizeof(float);
    std::cout << "Input dimensions: ";
    for (size_t i = 0; i < in_dims.dims.size(); ++i) {
        std::cout << in_dims.dims[i] << " ";
    }
    
    if (cudaMallocAsync(&d_input, input_bytes, stream) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input" << std::endl;
        return -1;
    }
    
    engine.preprocess(
        d_img,
        img.cols,
        img.rows,
        img.step,
        static_cast<float*>(d_input),
        1280,
        1280,
        true,
        stream
    );
    
    cudaStreamSynchronize(stream);
    
    // execute asynchronous inference. The return value just says whether the inference was successfully queued.
    if (!engine.execute(d_input, stream)) {
        std::cerr << "Failed to execute inference" << std::endl;
        return -1;
    }
    
    // Get the result. This function will return immediately, but the result will be available only after the inference is done. The result is a struct that contains pointers to the output tensors on the device, as well as their dimensions and the number of detections. You can either do postprocessing on the GPU using these pointers, or you can copy the results to the host and do postprocessing there. In this example we copy the results to the host for easier visualisation, but in a real application you probably want to do everything on the GPU.
    celantur::GPUInferenceResultDevice res = engine.get_result(stream);
    cudaStreamSynchronize(stream);
    
    // This is not important and just an example of how to extract the results and display them on the image.
    GPUInferenceResultHost2 host_res = GPUInferenceResultHost2::from_device_result(res, stream);
    auto detections = postprocess_injected2(host_res, cv::Size2i(img.cols, img.rows), cv::Size2i(1280, 1280));
    cv::Mat img_out = celantur::visualise_detections(img, detections);
    cv::imwrite(out_image_path, img_out);
}
