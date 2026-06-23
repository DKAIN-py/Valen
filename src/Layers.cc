#include"Layers.hpp"
#include<vector>
#include<string>
#include<fstream>

Linear::Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
    weights = Nexus({in_features, out_features});
    bias = Nexus({1,out_features});
}

void Linear::load_parameters(const std::string& weights_path, const std::string& bias_path){
    weights.load_from_binary(weights_path);
    bias.load_from_binary(bias_path);
}

Nexus Linear::forward(const Nexus& input){
    int batch_size = input.shape[0];
    Nexus output({batch_size, out_features});

    for(int b = 0; b<batch_size; ++b){
        for(int out = 0; out<out_features; ++out){
            float accumulator = 0.0f;

            for(int in = 0; in<in_features; ++in){
                int in_idx = input.get_index(b, in);
                int w_idx = weights.get_index(in, out);
                accumulator += input.data[in_idx]*weights.data[w_idx];
            }
            int bias_idx = bias.get_index(0, out);
            int out_idx = output.get_index(b, out);
            output.data[out_idx] = accumulator + bias.data[bias_idx];
        }
    }

    return output;
}