#include<iostream>
#include "Nexus.hpp"
#include<string>
#include"model_creation.hpp"
#include"models.hpp"

int main(int argc, char* argv[]){
    std::string dir_path = argv[1];

    CreateModel builder;
    Sequential model = builder.create_model(dir_path);
    Nexus input({1,4});    
    std::vector<float> value = {5.5 ,2.4 ,3.8 ,1.1};
    input.data = std::move(value);
    Nexus res = model.ForwardPass(input);
    
    for(const float ele : res.data){
        std::cout<<ele<<" ";
    }
    std::cout<<std::endl;
    return 0;
}