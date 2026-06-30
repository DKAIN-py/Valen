#include"models.hpp"

Nexus Sequential::ForwardPass(const Nexus& input){
        Nexus x = input;
        for(const auto& layer : this->list){
            x = layer->forward(x);
        }

        return x;
};