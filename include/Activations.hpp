#pragma once

#include "Nexus.hpp"
#include"BaseForward.hpp"

class ReLU : public BaseForward{
    public:
        ReLU() = default;
    
        Nexus forward(const Nexus& x) override;
};