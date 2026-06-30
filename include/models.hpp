#include"Nexus.hpp"
#include"BaseForward.hpp"
#include<iostream>
#include<vector>
#include<memory>

class Sequential{
    public:
        std::vector<std::unique_ptr<BaseForward>> list;
        Sequential() = default;

        
        Nexus ForwardPass(const Nexus& input);
};