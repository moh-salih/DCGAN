#ifndef DCGAN_UTILS_H
#define DCGAN_UTILS_H
#include "dcgan/config.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <assert.h>

namespace dcgan::utils{
    void preprocess(const cv::Mat& cvImage, torch::Tensor& torchImage);
    void postprocess(const torch::Tensor& torchImage, cv::Mat& cvImage);
    void imShow(const torch::Tensor& torchImage);
    void imSave(const torch::Tensor& torchImage, const std::string& filepath);
};
#endif //DCGAN_UTILS_H