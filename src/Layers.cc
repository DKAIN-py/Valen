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

Conv2D::Conv2D(std::vector<int> kernel_size, int in_channels, int out_channels, int stride, int pad_h, int pad_w, const Threadpool& pool_ref) : pool(pool_ref){
    this->kernel_size = kernel_size;
    this->in_channels = in_channels;
    this->stride = stride;
    this->pad_h = pad_h;
    this->pad_w = pad_w;
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

Nexus Conv2D::pad2d(const Nexus& input, int pad_h, int pad_w){
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int H_pad = H+2*pad_h;
    int W_pad = W+2*pad_w;

    const float* src = input.data.data();

    std::vector<float> dst(N*C*H_pad*W_pad, 0.0f);

    for(int n{0}; n<N; n++){
        for(int c{0}; c<C; c++){
            for(int h{0}; h<H; h++){
                const float* src_row = src +((n*C + c)*H + h)*W;
                float* dst_row = dst.data() + (((n*C + c)*H_pad + (h+pad_h))*W_pad) + pad_w;

                std::copy(src_row, src_row+W, dst_row);
            }
        }
    }
    Nexus padded_input({N,C,H_pad, W_pad});
    padded_input.data = std::move(dst);
    return padded_input;
    
}

Nexus Conv2D::im2col(const Nexus& input, int kh, int kw, int stride, int pad_h, int pad_w){
    Nexus padded_input = (pad_h>0 || pad_w>0) ? this->pad2d(input, pad_h, pad_w) : input;

    int N = padded_input.shape[0];
    int C = padded_input.shape[1];
    int H = padded_input.shape[2];
    int W = padded_input.shape[3];

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
    std::mdspan patches(padded_input.data.data(), patches_map);

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
    std::mdspan patches_transposed(padded_input.data.data(), trans_map);

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
    x_col.data = std::move(x_col_data);

    return x_col;
}



Nexus Conv2D::forward(const Nexus& input){
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    
    int kh = this->kernel_size[0];
    int kw = this->kernel_size[1];

    int H_out = (H + 2*this->pad_h - kh)/this->stride + 1;
    int W_out = (W + 2*this->pad_w - kw)/this->stride + 1;

    Nexus x_col = this->im2col(input, kh, kw, this->stride, this->pad_h, this->pad_w);

    Nexus output_mat({this->out_channels, N*H_out*W_out});

    int effective_batch_size{N*H_out*W_out};

    int threads = std::thread::hardware_concurrency();
    if(threads==0) threads = 4;
    int batches_per_thread = effective_batch_size/threads;

    std::vector<float>* output_data_ptr = &output_mat.data;

    std::latch barrier(threads);

    int in_features = C*kh*kw;
    int batch_cols = N*H_out*W_out;
    int out_features = out_channels;

    for(int i{0}; i<threads; i++){
        int start = i*batches_per_thread;
        int end = (i==threads-1) ? (effective_batch_size-1) : (start+batches_per_thread-1);

        this->pool.enqueue_task([this, &x_col, &input, &output_mat, start, end, &barrier, out_features, in_features,batch_cols]{
            const float* w_ptr = this->weights.data.data();
            const float* x_ptr = x_col.data.data();
            const float* b_ptr = this->bias.data.data();
            float* out_ptr = output_mat.data.data();

            for(int m{start}; m<=end; ++m){
                for(int out{0}; out < out_features; ++out){
                    float accumulator{0.0f};
                    const float* w_row = w_ptr + (out*in_features);

                    for(int in{0}; in<in_features; ++in){
                        accumulator += w_row[in]*x_ptr[in * batch_cols + m];
                    }

                    out_ptr[out*batch_cols+m] = accumulator + b_ptr[out];
                }
            }

            barrier.count_down();
        });
    }
    
    barrier.wait();
}