#include"models.hpp"

Nexus Sequential::ForwardPass(const Nexus& input){
        if(this->list.empty()) return input;
        
        Nexus x = this->list[0]->forward(input);
        for(int i = 1; i<this->list.size(); i++){
            x = this->list[i]->forward(x);
        }

        return x;
};

void Sequential::add(std::unique_ptr<BaseForward> layer){
        this->list.push_back(std::move(layer));
}