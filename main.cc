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
    ([&model](const crow::request& req) {
        std::cout << "Hit the home route\n";

        // 1. Check for the custom shape header
        std::string shape_header = req.get_header_value("X-Tensor-Shape");
        if (shape_header.empty()) {
            return crow::response(400, "Missing X-Tensor-Shape header");
        }

        // 2. Parse the dynamic shape string (e.g., "64,1,28,28" or "128,784" or "32,128,512")
        std::vector<int> shape;
        std::stringstream ss(shape_header);
        std::string token;
        size_t expected_elements = 1;

        try {
            while (std::getline(ss, token, ',')) {
                int dim = std::stoi(token);
                shape.push_back(dim);
                expected_elements *= dim;
            }
        } catch (const std::exception& e) {
            return crow::response(400, "Invalid X-Tensor-Shape format");
        }

        // 3. Verify payload size against expected float count
        size_t expected_bytes = expected_elements * sizeof(float);
        if (req.body.size() != expected_bytes) {
            std::cerr << "Payload size mismatch! Expected: " << expected_bytes 
                      << " bytes, Got: " << req.body.size() << " bytes\n";
            return crow::response(400, "Payload size does not match X-Tensor-Shape");
        }

        // 4. Construct Nexus tensor dynamically and memcpy raw floats
        Nexus input(shape);
        std::memcpy(input.data.data(), req.body.data(), expected_bytes);

        // 5. Run Forward Pass
        Nexus res = model.ForwardPass(input);

        // 6. Return response as JSON using nlohmann::json
        json response_data = res.get_ndim_data();
        
        crow::response crow_res(200, response_data.dump());
        crow_res.set_header("Content-Type", "application/json");
        return crow_res;
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