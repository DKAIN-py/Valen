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