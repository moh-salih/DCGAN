#include <iostream>
#include "dcgan/config.h"
#include "dcgan/utils.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "dcgan/DCGAN.h"
#include "dcgan/NatureDataset.h"

int main(int argc, char * argv[]){
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Using CPU." << std::endl;
        device = torch::kCPU;
    }

    try {
        const fs::path dataDir = dcgan::Config::DATA_DIR / "nature";
        auto dataset = dcgan::NatureDataset(dataDir)
                            .map(torch::data::transforms::Normalize(0.5, 0.5))
                            .map(torch::data::transforms::Stack<>());

        auto dataloader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(dcgan::Config::BATCH_SIZE));

        auto G = dcgan::Generator(dcgan::Config::LATENT_DIMENSION);
        auto D = dcgan::Discriminator();

        G->to(device);
        D->to(device);
        
        auto gOptimizer = torch::optim::Adam(G->parameters(), torch::optim::AdamOptions(dcgan::Config::LEARNING_RATE).betas({0.5, 0.5}));
        auto dOptimizer = torch::optim::Adam(D->parameters(), torch::optim::AdamOptions(dcgan::Config::LEARNING_RATE).betas({0.5, 0.5}));

        std::cout << std::fixed << std::setprecision(3);
        size_t batchIndex{};

        // Training
        for(size_t epoch{}; epoch < dcgan::Config::NUM_OF_EPOCHS; ++epoch) {
            for(auto& [realImages, IDoNotCareAboutLabels]: *dataloader) {
                const int64_t batchSize = realImages.size(0);

                // Move real images to the GPU
                realImages = realImages.to(device);

                // Train the Discriminator on real images.
                D->zero_grad();
                torch::Tensor realOutputs = D->forward(realImages);
                torch::Tensor realTargets = torch::ones_like(realOutputs).to(device);
                torch::Tensor dLossReal = torch::binary_cross_entropy(realOutputs, realTargets);
                dLossReal.backward(); // backpropagate for real images.

                // Train the Discriminator on fake images.
                torch::Tensor z = torch::randn({batchSize, dcgan::Config::LATENT_DIMENSION, 1, 1}).to(device);
                torch::Tensor fakeImages = G->forward(z);
                torch::Tensor fakeOutputs = D->forward(fakeImages.detach());
                torch::Tensor fakeTargets = torch::zeros_like(fakeOutputs).to(device);
                torch::Tensor dLossFake = torch::binary_cross_entropy(fakeOutputs, fakeTargets);
                dLossFake.backward(); // backpropagate for fake images.

                // Update Discriminator parameters.
                dOptimizer.step();

                torch::Tensor dLoss = dLossReal + dLossFake;

                // Train the generator
                G->zero_grad();
                fakeOutputs = D->forward(fakeImages);
                torch::Tensor gTargets = torch::ones_like(fakeOutputs).to(device);
                torch::Tensor gLoss = torch::binary_cross_entropy(fakeOutputs, gTargets);
                
                gLoss.backward();
                gOptimizer.step();
                
                batchIndex++;

                // Save images and print losses every 5 batches
                if(batchIndex % 5 == 0) {
                    dcgan::utils::imSave(fakeImages[0].cpu(), 
                    (dcgan::Config::IMAGE_DIR / std::to_string(epoch).append("_").append(std::to_string(batchIndex).append(".png"))).string());

                    std::cout << "Epoch [" << epoch << "/" << dcgan::Config::NUM_OF_EPOCHS << "], Batch [" << batchIndex << "]: "
                        << "G loss: " << gLoss.item<float>() 
                        << ", D loss: " << dLoss.item<float>()
                        << ", Real loss: " << dLossReal.item<float>() 
                        << ", Fake loss: " << dLossFake.item<float>() 
                        << ", Real score: " << realOutputs.mean().item<double>() 
                        << ", Fake score: " << fakeOutputs.mean().item<double>() << '\n'; 
                }
            }

            if(epoch % 10 == 0){
                torch::save(G, (dcgan::Config::MODEL_DIR / std::to_string(epoch).append(".pt")).string());
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error/Exception: " << e.what() << '\n';
    }
    
    return 0;
}
