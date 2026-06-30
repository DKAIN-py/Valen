#include "Nexus.hpp"
#include"BaseForward.hpp"

class ReLU : public BaseForward{
    ReLU() = default;

    Nexus forward(const Nexus& x) override;
};