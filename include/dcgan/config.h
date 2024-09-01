#ifndef DCGAN_CONFIG_H
#define DCGAN_CONFIG_H
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace dcgan::Config{
    const size_t BATCH_SIZE = 32;
    const size_t NUM_OF_EPOCHS = 200;
    const size_t LATENT_DIMENSION = 100;
    const float LEARNING_RATE = 2e-4;

    const cv::Size IMAGE_SIZE(1280, 720); 


    const fs::path ROOT_DIR = fs::path(__FILE__).parent_path().parent_path().parent_path(); // Assuming that this file will always stay where it is.
    const fs::path DATA_DIR = ROOT_DIR / "data" / "input";
    const fs::path IMAGE_DIR = ROOT_DIR / "data" / "working" / "images";
    const fs::path MODEL_DIR = ROOT_DIR / "data" / "working" / "models";
};

#endif // DCGAN_CONFIG_H