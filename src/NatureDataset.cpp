#include "dcgan/NatureDataset.h"
#include "dcgan/utils.h"
#include <opencv2/opencv.hpp>
#include <assert.h>

namespace dcgan{
    NatureDataset::NatureDataset(const fs::path& dataDir){
        assert(fs::exists(dataDir));
        loadImages(dataDir);
    }

    void NatureDataset::loadImages(const fs::path& dataDir){
        for(const auto& entry: fs::directory_iterator(dataDir)){
            if(entry.is_regular_file()){
                imagePaths.push_back(entry.path().string());
            }
        }
    }

    torch::data::Example<> NatureDataset::get(size_t index){
        auto imagePath = imagePaths.at(index);
        cv::Mat cvImage = cv::imread(imagePath);
        
        torch::Tensor torchImage;
        utils::preprocess(cvImage.clone(), torchImage);

        return { torchImage, torch::tensor(static_cast<int64_t>(index), torch::kInt64)};
    }
};
