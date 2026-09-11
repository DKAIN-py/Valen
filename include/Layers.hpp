#pragma once

// Manual Headers
#include "Nexus.hpp"
#include"BaseForward.hpp"
#include"Threadpool.hpp"

class Linear : public BaseForward{
    private:
        const Threadpool& pool;
    public:
        Nexus weights;
        Nexus bias;
        int in_features;
        int out_features;

        Linear(int in_f, int out_f, const Threadpool& pool_ref);

        void load_parameters(const std::string& weights_path, const std::string& bias_path);

        Nexus forward(const Nexus& input) override;
};

class Conv2D : public BaseForward{
    private:
        const Threadpool& pool;
    public:
        std::vector<int> kernel_size;
        Nexus weights;
        Nexus bias;
        int in_channels;
        int out_channels;
        int stride{1};
        int padding{0};

        Conv2D(std::vector<int> kernel_size, int in_channels, int out_channels, int stride, int padding, const Threadpool& pool_ref);

        void load_parameters(const std::string& weights_path, const std::string& bias_path);

        Nexus im2col(Nexus& input, int kh, int kw, int stride, int padding);

        Nexus forward(const Nexus& input) override;
};

class MaxPool2D : public BaseForward{
    public:
        std::vector<int> kernel_size;
        int stride;

        MaxPool2D(std::vector<int> kernel_size, int stride);

        Nexus forward(const Nexus& input) override;
};