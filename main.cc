#include<iostream>
#include "Nexus.hpp"
#include<string>
#include"model_creation.hpp"
#include"models.hpp"

int main(int argc, char* argv[]){
    std::string dir_path = argv[1];

    CreateModel builder;
    Sequential model = builder.create_model(dir_path);
    Nexus input({20,4});    
    std::vector<std::vector<float>> value = {
        {5.5, 2.4, 3.8, 1.1},
        {5.7, 2.6, 3.5, 1.0},
        {5.0, 3.0, 1.6, 0.2},
        {5.7, 4.4, 1.5, 0.4},
        {5.1, 3.4, 1.5, 0.2},
        {5.0, 3.5, 1.3, 0.3},
        {5.5, 2.3, 4.0, 1.3},
        {5.1, 3.5, 1.4, 0.3},
        {4.9, 2.4, 3.3, 1.0},
        {7.0, 3.2, 4.7, 1.4},
        {5.7, 3.8, 1.7, 0.3},
        {5.6, 2.5, 3.9, 1.1},
        {6.3, 2.3, 4.4, 1.3},
        {5.4, 3.9, 1.3, 0.4},
        {4.6, 3.2, 1.4, 0.2},
        {4.6, 3.1, 1.5, 0.2},
        {4.4, 3.0, 1.3, 0.2},
        {6.9, 3.1, 4.9, 1.5},
        {4.4, 2.9, 1.4, 0.2},
        {4.9, 3.1, 1.5, 0.1}
    };

    input.load_input(value);
    Nexus res = model.ForwardPass(input);
    
    for(const float ele : res.data){
        std::cout<<ele<<" ";
    }
    std::cout<<std::endl;
    return 0;
}