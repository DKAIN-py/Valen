#define CROW_MAIN

#include<iostream>
#include "Nexus.hpp"
#include<string>
#include"model_creation.hpp"
#include"models.hpp"
#include<fstream>
#include"Threadpool.hpp"
#include"crow_all.h"

using json = nlohmann::json;

int main(int argc, char* argv[]){
    crow::SimpleApp app;

    std::string dir_path = argv[1];

    size_t threads = std::thread::hardware_concurrency();
    Threadpool pool(threads);
    
    CreateModel builder;
    Sequential model = builder.create_model(dir_path, pool);
    
    CROW_ROUTE(app, "/").methods(crow::HTTPMethod::POST)
    ([&model](const crow::request& req){
        std::cout<<"Hit the home route "<<"\n";
        
        json json_data;

        try{
            json_data = json::parse(req.body);

        }catch(const json::parse_error& e){
            std::cerr<<"JSON Parsing error:"<<e.what()<<std::endl;
            return crow::response(400, "Invalid JSON payload");
        }
    
        std::vector<std::vector<float>> value = json_data.get<std::vector<std::vector<float>>>();
        
        int batch_size = static_cast<int>(value.size());
        int features = static_cast<int>(value[0].size());
        Nexus input({batch_size, features});    
        
        input.load_input(value);
        Nexus res = model.ForwardPass(input);
        json response_data = res.get_ndim_data();

        return crow::response(200, response_data.dump());
    });

    app.port(8080).run();


    // std::ifstream file("iris_dummy_data.json");
    //     if (!file.is_open()) {
    //         std::cerr << "Error: Could not open the JSON file!" << std::endl;
    //         return 1;
    //     }

    // try {
    //     file >> json_data;
    // } catch (const json::parse_error& e) {
    //     std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
    //     return 1;
    // }
    
    // file.close();

    
    
    
    // for(const float ele : res.data){
    //     std::cout<<ele<<" ";
    // }
    // std::cout<<std::endl;
    return 0;
}