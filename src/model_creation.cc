#include"model_creation.hpp"
#include"json.hpp"
#include<fstream>
#include<optional>
#include<memory>
#include"Layers.hpp"
#include"Activations.hpp"
#include<stdexcept>

using json = nlohmann::json;

Sequential CreateModel::create_model(const std::string& input, const Threadpool& pool_ref){
        std::string manifest_path = input+"/model_manifest.json";

        std::ifstream file(manifest_path);
        json manifest_file;
        if(file.is_open()){
            file >> manifest_file;
        }

        auto model_manifest = manifest_file.get<std::vector<json_helper::LayerObject>>();
        
        Sequential model;
        for(const auto& obj : model_manifest){
            
            if(obj.type=="Linear"){
            
                int in_f = obj.shape_W.value()[0];
                int out_f = obj.shape_W.value()[1];
                std::string w_file = input+"/"+obj.weights_file.value();
                std::string b_file = input+"/"+obj.bias_file.value();
                
                std::unique_ptr<Linear> l1 = std::make_unique<Linear>(in_f, out_f, pool_ref);
                l1->load_parameters(w_file, b_file);
                
                model.add(std::move(l1));
            }

            else if(obj.type=="Conv2D"){
                int w_h = obj.shape_W.value()[0];
                int w_w = obj.shape_W.value()[1];
                int b_h = obj.shape_B.value()[0];
                int b_w = obj.shape_B.value()[1];
                std::string w_file = input+"/"+obj.weights_file.value();
                std::string b_file = input+"/"+obj.bias_file.value();
                std::vector<int> kernel_size = obj.kernel_size.value();
                int padding = obj.padding.value();
                int in_channels = obj.in_channels.value();
                int out_channels = obj.out_channels.value();
                int stride = obj.stride.value();

                std::unique_ptr<Conv2D> c1 = std::make_unique<Conv2D>(
                    kernel_size, in_channels, out_channels, stride,
                    padding, padding, pool_ref
                );

                c1->load_parameters(w_file, b_file);

                model.add(std::move(c1));
            }

            else if(obj.type=="MaxPool2D"){
                std::vector<int> kernel_size = obj.kernel_size.value();
                int stride = obj.stride.value();

                std::unique_ptr<MaxPool2D> m = std::make_unique<MaxPool2D>(kernel_size, stride);

                model.add(std::move(m));
            }

            else if(obj.type=="Flatten"){
                model.add(std::make_unique<Flatten>());
            }

            else if(obj.type=="ReLU"){
                model.add(std::make_unique<ReLU>());
            }
            else{
                std::string err = "Layer type "+ obj.type +" dosen't exists in the program.\n";
                throw std::invalid_argument(err);
            }
        }
        
        return model;
    }