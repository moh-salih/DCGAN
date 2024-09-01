#include "dcgan/DCGAN.h"
#include "dcgan/config.h"

namespace dcgan{
    // Generator implementation.
    GeneratorImpl::GeneratorImpl(int kNoiseSize): 
        conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 1024, 4).bias(false)),
        batch_norm1(1024),
        conv2(torch::nn::ConvTranspose2dOptions(1024, 512, 4).stride(2).padding(1).bias(false)),
        batch_norm2(512),
        conv3(torch::nn::ConvTranspose2dOptions(512, 256, 4).stride(2).padding(1).bias(false)),
        batch_norm3(256),
        conv4(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1).bias(false)),
        batch_norm4(128),
        conv5(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
        batch_norm5(64),
        conv6(torch::nn::ConvTranspose2dOptions(64, 3, 4).stride(2).padding(1).bias(false)) 
    {
        register_module("conv1", conv1);
        register_module("batch_norm1", batch_norm1);
        register_module("conv2", conv2);
        register_module("batch_norm2", batch_norm2);
        register_module("conv3", conv3);
        register_module("batch_norm3", batch_norm3);
        register_module("conv4", conv4);
        register_module("batch_norm4", batch_norm4);
        register_module("conv5", conv5);
        register_module("batch_norm5", batch_norm5);
        register_module("conv6", conv6);
    }

    torch::Tensor GeneratorImpl::forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::relu(batch_norm4(conv4(x)));
        x = torch::relu(batch_norm5(conv5(x)));
        x = torch::tanh(conv6(x));
        return x;
    }


    // Discriminator implementation.
    DiscriminatorImpl::DiscriminatorImpl(): 
        conv1(torch::nn::Conv2dOptions(3, 64, 4).stride(2).padding(1).bias(false)),
        conv2(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        batch_norm2(128),
        conv3(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        batch_norm3(256),
        conv4(torch::nn::Conv2dOptions(256, 512, 4).stride(2).padding(1).bias(false)),
        batch_norm4(512),
        conv5(torch::nn::Conv2dOptions(512, 1024, 4).stride(2).padding(1).bias(false)),
        batch_norm5(1024),
        conv6(torch::nn::Conv2dOptions(1024, 1, 4).stride(1).padding(0).bias(false))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("batch_norm2", batch_norm2);
        register_module("conv3", conv3);
        register_module("batch_norm3", batch_norm3);
        register_module("conv4", conv4);
        register_module("batch_norm4", batch_norm4);
        register_module("conv5", conv5);
        register_module("batch_norm5", batch_norm5);
        register_module("conv6", conv6);
    }

    torch::Tensor DiscriminatorImpl::forward(torch::Tensor x) {
        x = torch::leaky_relu(conv1(x), 0.2);
        x = torch::leaky_relu(batch_norm2(conv2(x)), 0.2);
        x = torch::leaky_relu(batch_norm3(conv3(x)), 0.2);
        x = torch::leaky_relu(batch_norm4(conv4(x)), 0.2);
        x = torch::leaky_relu(batch_norm5(conv5(x)), 0.2);
        x = torch::sigmoid(conv6(x));
        return x.view({-1, 1});
    }
};