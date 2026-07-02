#pragma once

#include "models.hpp"
#include<optional>
#include<vector>
#include"json.hpp"
#include"Layers.hpp"
#include"Activations.hpp"

using json = nlohmann::json;

namespace json_helper {
    
    struct LayerObject{
        // required
        unsigned index;
        std::string type;
        
        // optional
        std::optional<std::string> weights_file;
        std::optional<std::string> bias_file;
        std::optional<std::vector<int>> shape_W;
        std::optional<std::vector<int>> shape_B;
    };

    void inline from_json(const json& j, LayerObject& layer){
        // required
        j.at("index").get_to(layer.index);
        j.at("type").get_to(layer.type);

        // optional
        layer.weights_file = j.contains("weights_file") ?std::make_optional(j["weights_file"].get<std::string>())       :std::nullopt;
        layer.bias_file    = j.contains("bias_file")    ?std::make_optional(j["bias_file"].get<std::string>())          :std::nullopt;
        layer.shape_W      = j.contains("shape_W")      ?std::make_optional(j["shape_W"].get<std::vector<int>>())  :std::nullopt;
        layer.shape_B      = j.contains("shape_B")      ?std::make_optional(j["shape_B"].get<std::vector<int>>())  :std::nullopt;

    };
}


class CreateModel{
    public:
        CreateModel() = default;

        Sequential create_model(const std::string& input);
};

