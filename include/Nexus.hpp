#pragma once
#include<vector>
#include<string>

class Nexus {
    public:
        std::vector<float> data;
        std::vector<int> shape;
        std::vector<int> strides;

        Nexus(std::vector<int> tensor_shape);

        Nexus() = default;

        void load_from_binary(const std::string& filepath);

        void load_input(const std::vector<std::vector<float>>& input);

        void update_strides();

        inline int get_index(int row, int col) const {
            return row*shape.at(1) + col;
        }

        inline int get_index(const std::vector<int>& coords) const {
            int flat_idx = 0;
            for(size_t i = 0; i<coords.size(); i++) flat_idx += coords[i]*strides[i];

            return flat_idx;
        }
};  