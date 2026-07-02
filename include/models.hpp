#pragma once

#include"Nexus.hpp"
#include"BaseForward.hpp"
#include<iostream>
#include<vector>
#include<memory>

class Sequential{
    private:
            std::vector<std::unique_ptr<BaseForward>> list;

    public:
        Sequential() = default;
        
        Sequential(Sequential&&) = default;
        Sequential& operator=(Sequential&) = default;

        Sequential(const Sequential&) = delete;
        Sequential& operator=(const Sequential&) = delete;

        Nexus ForwardPass(const Nexus& input);
        
        void add(std::unique_ptr<BaseForward> layer);
};