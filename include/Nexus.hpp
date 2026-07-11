#pragma once
#include<vector>
#include<string>

class Nexus {
    public:
        std::vector<float> data;
        std::vector<int> shape;

        Nexus(std::vector<int> tensor_shape);

        Nexus() = default;

        void load_from_binary(const std::string& filepath);

        void load_input(const std::vector<std::vector<float>>& input);

        inline int get_index(int row, int col) const {
            return row*shape[1] + col;
        }
};  