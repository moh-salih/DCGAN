#ifndef DCGAN_H
#define DCGAN_H
#include <torch/torch.h>

namespace dcgan{
    struct GeneratorImpl : torch::nn::Module {
        GeneratorImpl(int kNoiseSize);
        torch::Tensor forward(torch::Tensor x);

        torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5, conv6;
        torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4, batch_norm5;
    };
    TORCH_MODULE(Generator);

    struct DiscriminatorImpl : torch::nn::Module {
        DiscriminatorImpl();
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5, conv6;
        torch::nn::BatchNorm2d batch_norm2, batch_norm3, batch_norm4, batch_norm5;
    };
    TORCH_MODULE(Discriminator);
};
#endif // DCGAN_H