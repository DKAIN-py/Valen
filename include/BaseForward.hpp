#pragma once
#include "Nexus.hpp"

class BaseForward{
    public:
        virtual ~BaseForward() = default;

        virtual Nexus forward(const Nexus& input) = 0;

        virtual ~BaseForward() {}
};