#pragma once
#include "Nexus.hpp"
#include<string>

class Linear {
    public:
        Nexus weights;
        Nexus bias;
        int in_features;
        int out_features;

        Linear(int in_f, int out_f);

        void load_parameters(const std::string& weights_path, const std::string& bias_path);

        Nexus forward(const Nexus& input);
};