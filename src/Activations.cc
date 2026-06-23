#include "Activations.hpp"
#include "Nexus.hpp"

Nexus ReLU::forward(const Nexus& x){
    Nexus output(x.shape);

    for(size_t i = 0; i<x.data.size(); i++){
        output.data[i] = std::max(0.0f, x.data[i]);
    }

    return output;
}