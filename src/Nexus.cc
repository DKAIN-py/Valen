#include "Nexus.hpp"
#include<fstream>
#include<iostream>

Nexus::Nexus(std::vector<int> tensor_shape) : shape(tensor_shape) {
    size_t total_element = 1;
    for(int dim : shape) total_element*=dim;

    data.resize(total_element, 0.0f);
}

void Nexus::load_from_binary(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if(!file.is_open()){
        std::cerr<<"Valen Runtime Error: Failed to open binary parameters "<<filepath<<std::endl;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected_bytes = data.size()*sizeof(float);
    if(static_cast<size_t>(file_size) != expected_bytes){
        std::cerr<<"Size Mismatch "<<filepath<<
        "Expected "<<expected_bytes<<" bytes, but got "<<file_size<<" bytes."<<std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(data.data()), expected_bytes);
    file.close();
    std::cout<<"Mapped Continous Memory buffer from "<<filepath<<" ["<<data.size()<<" elements]"<<std::endl;
}

void Nexus::load_input(const std::vector<std::vector<float>>& input){
    if(input.empty() || input[0].empty()){
        shape = {0,0};
        return;
    }
    int batch_size = static_cast<int>(input.size());
    int features = static_cast<int>(input[0].size());
    data.clear();
    data.reserve(batch_size*features);

    for(const auto& row : input){
        if(static_cast<int>(row.size()) != features){
            std::cerr<<"Valen Runtime Error: All the rows in input matrix should be uniform!";
            return;
        }

        data.insert(data.end(), row.begin(), row.end());
    };
}

