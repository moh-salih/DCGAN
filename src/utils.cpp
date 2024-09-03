#include "dcgan/utils.h"

namespace dcgan::utils{

    void preprocess(const cv::Mat& cvImage, torch::Tensor& torchImage){
        cv::Mat image = cvImage.clone();

        // Make sure we have a valid image.
        assert(!image.empty());

        // Resizing
        cv::resize(image, image, Config::IMAGE_SIZE);
        
        // Convert the image to a tensor.
        torchImage = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kUInt8);


        // Permute and convert map range to [0-1] assuming the image is stored as 8-byte unsigned char.
        torchImage = torchImage.permute({2, 0, 1});
        torchImage = torchImage.to(torch::kFloat32);
        torchImage = torchImage.div(255).clone();
    }

    void postprocess(const torch::Tensor& torchImage, cv::Mat& cvImage){
        auto image = torchImage.clone().detach();
        image = image.permute({1, 2, 0}).clone(); // convert back to HxWxC
        image = image.contiguous();
        image = image.mul(0.5).add(0.5); // denormalize
        image = image.mul(255).clamp(0, 255).to(torch::kUInt8); // map to [0-255] range and convert back to CV_8UC3.
        
        cvImage = cv::Mat(image.size(0), image.size(1), CV_8UC3, image.data_ptr<uint8_t>()).clone();
    }

    void imShow(const torch::Tensor& torchImage){
        std::cout << "Received a torch image in with the shape: " << torchImage.sizes() << '\n';
        cv::Mat cvImage;
        dcgan::utils::postprocess(torchImage, cvImage);
        cv::imshow("window", cvImage);
        cv::waitKey();
    }

    void imSave(const torch::Tensor& torchImage, const std::string& filepath){
        cv::Mat cvImage;
        dcgan::utils::postprocess(torchImage, cvImage);
        cv::imwrite(filepath, cvImage);
    }

}