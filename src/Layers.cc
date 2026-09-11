// Manual Header
#include"Layers.hpp"

// Standard Headers
#include<chrono>
#include<iostream>
#include<latch>
#include<mdspan>
#include<array>

Linear::Linear(int in_f, int out_f, const Threadpool& pool_ref) : in_features(in_f), out_features(out_f), pool(pool_ref){
    weights = Nexus({in_features, out_features});
    bias = Nexus({1,out_features});
}

void Linear::load_parameters(const std::string& weights_path, const std::string& bias_path){
    weights.load_from_binary(weights_path);
    bias.load_from_binary(bias_path);
}

Nexus Linear::forward(const Nexus& input){
    int effective_batch_size = 1;
    for(int i = 0; i< input.shape.size()-1; i++) effective_batch_size*=input.shape[i];
    // int batch_size = input.shape.at(0);
    
    std::vector<int> out_shape = input.shape;
    out_shape.back() = out_features;
    
    Nexus output(out_shape);
    
    // Compute kernel performence monitoring
    auto start_comput = std::chrono::high_resolution_clock::now();

    if(effective_batch_size<=100){
        for(int b = 0; b<effective_batch_size; ++b){
            for(int out = 0; out<out_features; ++out){
                float accumulator = 0.0f;

                for(int in = 0; in<in_features; ++in){

                    int in_batch_offset = b * in_features;
                    int in_idx = in_batch_offset + in;
                    int w_idx = weights.get_index(in, out);
                    accumulator += input.data.at(in_idx)*weights.data.at(w_idx);
                
                }
                
                int bias_idx = bias.get_index(0, out);
                int out_batch_offset = b * out_features;
                int out_idx = out_batch_offset + out;
                output.data.at(out_idx) = accumulator + bias.data.at(bias_idx);
            }
        }
    }

    else{
        
        int threads = std::thread::hardware_concurrency();
        if (threads==0) threads=4;

        int batches_per_thread = effective_batch_size/threads;
        
        std::vector<float> *output_data_ptr = &output.data;

        std::latch barrier(threads);

        for(int i = 0; i<threads; i++){
            
            int start = i*batches_per_thread;
            int end = (i == threads - 1) ? (effective_batch_size - 1) : (start + batches_per_thread - 1);
            
            this->pool.enqueue_task([this, &input, &output,  start, end, &barrier] {
                for(int b = start; b<=end; ++b){
                    int out_batch_offset = b * out_features;
                    int in_batch_offset = b * in_features;

                    for(int out = 0; out<out_features; ++out){
                        float accumulator = 0.0f;
                    
                        for(int in = 0; in<in_features; ++in){
                            int in_idx = in_batch_offset + in;
                            int w_idx = weights.get_index(in, out);

                            accumulator += input.data.at(in_idx)*weights.data.at(w_idx);
                        }

                        int bias_idx = bias.get_index(0, out);
                        int out_idx = out_batch_offset + out;
                                        
                        output.data.at(out_idx) = accumulator + bias.data.at(bias_idx);
                    }
                }
                barrier.count_down();
            });
            
        }

        barrier.wait();
    }
    
    
    auto end_comput = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> compute_time_ms = end_comput - start_comput;
    std::cout << "\n========================================\n";
    std::cout << "VALEN COMPUTE KERNEL METRICS:\n";
    std::cout << "   Strict Calculation Time: " << compute_time_ms.count() << " ms\n";
    std::cout << "========================================\n";


    return output;
}


// 2D Convlution

Conv2D::Conv2D(std::vector<int> kernel_size, int in_channels, int out_channels, int stride, int padding, const Threadpool& pool_ref) : pool(pool_ref){
    this->kernel_size = kernel_size;
    this->in_channels = in_channels;
    this->stride = stride;
    this->padding = padding;
    this->weights = Nexus({out_channels, in_channels*kernel_size[0]*kernel_size[1]});
    this->bias = Nexus({out_channels, 1});
}

void Conv2D::load_parameters(const std::string& weights_path, const std::string& bias_path){
    weights.load_from_binary(weights_path);
    bias.load_from_binary(bias_path);
}

using Dynamic6DExtents = std::extents<int,
    std::dynamic_extent, // N
    std::dynamic_extent, // C
    std::dynamic_extent, // H_out
    std::dynamic_extent, // W_out
    std::dynamic_extent, // kh
    std::dynamic_extent>; // kw


Nexus Conv2D::im2col(Nexus& input, int kh, int kw, int stride, int padding){
    if (padding > 0){
        // input.padding(padding);
    }

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int H_out = (H-kh)/stride + 1;
    int W_out = (W-kw)/stride + 1;

    int x_stride_3{1};
    int x_stride_2{W};
    int x_stride_1{H*W};
    int x_stride_0{C*H*W};

    // kernel window over 4D tensor to array of all the windows 
    Dynamic6DExtents patches_shape(N,C,H_out,W_out,kh,kw);
    std::array<int, 6> patches_stride = {
        x_stride_0,
        x_stride_1,
        x_stride_2*stride,
        x_stride_3*stride,
        x_stride_2,
        x_stride_3
    };

    std::layout_stride::mapping patches_map(patches_shape, patches_stride);
    std::mdspan patches(input.data.data(), patches_map);

    // transposed tensor 
    Dynamic6DExtents trans_shape(
        patches.extent(1), // C
        patches.extent(4), // kh
        patches.extent(5), // kw
        patches.extent(0), // N
        patches.extent(2), // H_out
        patches.extent(3)  // W_out
    );

    std::array<int, 6> trans_stride = {
        patches_stride[1], // C stride
        patches_stride[4], // kh stride
        patches_stride[5], // kw stride
        patches_stride[0], // N
        patches_stride[2], // H_out stride
        patches_stride[3]  // W_out stride
    };

    std::layout_stride::mapping trans_map(trans_shape, trans_stride);
    std::mdspan patches_transposed(input.data.data(), trans_map);

    // reshpae to 2D matrix for GEMM
    int rows = C*kh*kw;
    int cols = N*H_out*W_out;

    std::vector<float> x_col_data(rows*cols);
    int linear_idx{0};

    for (int d0 = 0; d0 < patches_transposed.extent(0); ++d0) {       // C
        for (int d1 = 0; d1 < patches_transposed.extent(1); ++d1) {   // kh
            for (int d2 = 0; d2 < patches_transposed.extent(2); ++d2) { // kw
                for (int d3 = 0; d3 < patches_transposed.extent(3); ++d3) { // N
                    for (int d4 = 0; d4 < patches_transposed.extent(4); ++d4) { // H_out
                        for (int d5 = 0; d5 < patches_transposed.extent(5); ++d5) { // W_out
                            
                            x_col_data[linear_idx++] = patches_transposed[d0, d1, d2, d3, d4, d5];
                            
                        }
                    }
                }
            }
        }
    }

    Nexus x_col({rows, cols});
    x_col.data = x_col_data;

    return x_col;
}