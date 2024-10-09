#include "Processor.h"
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char* argv[])
{
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

    celantur::Processor processor(model_path);
    processor.create_execution_context();
    cv::Mat image_bgr = cv::imread(img_path);
    cv::Mat image;
    cv::cvtColor(image_bgr, image, cv::COLOR_BGR2RGB);
    processor.process(image);
    cv::Mat output = processor.get_next();
    processor.get_detections();
    cv::Mat output_bgr;
    cv::cvtColor(output, output_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite("output.jpg", output_bgr);

    return 0;
}