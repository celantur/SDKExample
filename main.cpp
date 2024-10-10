#include "Processor.h"
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char* argv[])
{
    // Parse input parameters
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <path to file> <path to model>" << std::endl ;
        return 1;
    }

    std::string img_path = argv[1];
    if (!std::filesystem::exists(img_path))
    {
        std::cout << "File not found: " << img_path << std::endl;
        return 1;
    }

    std::string model_path = argv[2];
    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Model not found: " << model_path << std::endl;
        return 1;
    }

    const int horizontal_tiles = 2, vertical_tiles = 2;
    const float horizontal_overlap = 0.1, vertical_overlap = 0.1;

    // Create processor with given tiling parameters
    celantur::Processor processor(model_path, horizontal_tiles, vertical_tiles,
                                  horizontal_overlap, vertical_overlap);

    // Create execution context; if your install does not support dynamic models, do not change the default parameters 
    // for this method
    processor.create_execution_context();

    // Read image and conver OpenCV default BGR to RGB
    // There are other more cheap ways to construct matrix from the given buffer,
    // See OpenCV documentation for more details
    cv::Mat image_bgr = cv::imread(img_path);
    cv::Mat image;
    cv::cvtColor(image_bgr, image, cv::COLOR_BGR2RGB);

    // Send image to processing queue
    processor.process(image);

    // Get the processed image from processing queue
    cv::Mat output = processor.get_next();
    // Get detections from the processing queue; Needs to be done each time you get the processed image
    // Otherwise detections will reside in the queue and the memory with steadily grow
    processor.get_detections();

    // Convert RGB back to BGR to save the image with OpenCV and save it
    cv::Mat output_bgr;
    cv::cvtColor(output, output_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite("output.jpg", output_bgr);

    return 0;
}