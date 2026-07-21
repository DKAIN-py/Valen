// Manual Header
#include"Layers.hpp"

// Standard Headers
#include<chrono>
#include<iostream>
#include<latch>

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
    for(size_t i = 0; i< input.shape.size()-1; i++) effective_batch_size*=input.shape[i];
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
        
        size_t threads = std::thread::hardware_concurrency();
        if (threads==0) threads=4;

        size_t batches_per_thread = effective_batch_size/threads;
        
        std::vector<float> *output_data_ptr = &output.data;

        std::latch barrier(threads);

        for(size_t i = 0; i<threads; i++){
            
            size_t start = i*batches_per_thread;
            size_t end = (i == threads - 1) ? (effective_batch_size - 1) : (start + batches_per_thread - 1);
            
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