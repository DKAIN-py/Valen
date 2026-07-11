#include<iostream>
#include "Nexus.hpp"
#include<string>
#include"model_creation.hpp"
#include"models.hpp"
#include<fstream>

using json = nlohmann::json;

int main(int argc, char* argv[]){
    std::string dir_path = argv[1];

    CreateModel builder;
    Sequential model = builder.create_model(dir_path);
    
    std::ifstream file("iris_dummy_data.json");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open the JSON file!" << std::endl;
            return 1;
        }

    json json_data;
    try {
        file >> json_data;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
        return 1;
    }
    
    file.close();

    std::vector<std::vector<float>> value = json_data.get<std::vector<std::vector<float>>>();
    
    int batch_size = static_cast<int>(value.size());
    int features = static_cast<int>(value[0].size());
    Nexus input({batch_size, features});    

    input.load_input(value);
    Nexus res = model.ForwardPass(input);
    
    // for(const float ele : res.data){
    //     std::cout<<ele<<" ";
    // }
    std::cout<<std::endl;
    return 0;
}