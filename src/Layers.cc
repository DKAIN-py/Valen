#include"Layers.hpp"
#include<vector>
#include<string>
#include<fstream>
#include<chrono>
#include<iostream>
#include"Threadpool.hpp"
#include<atomic>
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
    int batch_size = input.shape.at(0);
    Nexus output({batch_size, out_features});
    std::printf("Output data object size : %d\n", output.data.size());
    // Compute kernel performence monitoring
    auto start_comput = std::chrono::high_resolution_clock::now();

    if(batch_size<=100){
        for(int b = 0; b<batch_size; ++b){
            for(int out = 0; out<out_features; ++out){
                float accumulator = 0.0f;

                for(int in = 0; in<in_features; ++in){
                    int in_idx = input.get_index(b, in);
                    int w_idx = weights.get_index(in, out);
                    accumulator += input.data.at(in_idx)*weights.data.at(w_idx);
                }
                int bias_idx = bias.get_index(0, out);
                int out_idx = output.get_index(b, out);
                output.data.at(out_idx) = accumulator + bias.data.at(bias_idx);
            }
        }
    }

    else{
        
        size_t threads = std::thread::hardware_concurrency();
        if (threads==0) threads=4;

        size_t batches_per_thread = batch_size/threads;
        
        std::vector<float> *output_data_ptr = &output.data;

        // std::atomic<size_t> active_task{threads};
        std::latch barrier(threads);

        for(size_t i = 0; i<threads; i++){
            
            size_t start = i*batches_per_thread;
            size_t end = (i == threads - 1) ? (batch_size - 1) : (start + batches_per_thread - 1);
            
            this->pool.enqueue_task([this, input, &output,  start, end, &barrier] {
                for(int b = start; b<=end; ++b){
                    for(int out = 0; out<out_features; ++out){
                        float accumulator = 0.0f;
                    
                        for(int in = 0; in<in_features; ++in){
                            int in_idx = input.get_index(b, in);
                            int w_idx = weights.get_index(in, out);

                            // Manual Safety Check for Input
                            if (in_idx < 0 || in_idx >= input.data.size()) {
                                std::printf("\n[CRASH] input.data out of bounds! Index: %d, Size: %lu\n", in_idx, input.data.size());
                                std::printf("Context: batch=%lu, in=%d, out=%d\n", b, in, out);
                                std::exit(1);
                            }
                            // Manual Safety Check for Weights
                            if (w_idx < 0 || w_idx >= weights.data.size()) {
                                std::printf("\n[CRASH] weights.data out of bounds! Index: %d, Size: %lu\n", w_idx, weights.data.size());
                                std::printf("Context: batch=%lu, in=%d, out=%d\n", b, in, out);
                                std::exit(1);
                            }

                            accumulator += input.data.at(in_idx)*weights.data.at(w_idx);
                        }
                        int bias_idx = bias.get_index(0, out);
                        int out_idx = output.get_index(b, out);
                        
                        // Manual Safety Check for Bias
                        if (bias_idx < 0 || bias_idx >= bias.data.size()) {
                            std::printf("\n[CRASH] bias.data out of bounds! Index: %d, Size: %lu\n", bias_idx, bias.data.size());
                            std::exit(1);
                        }
                        // Manual Safety Check for Output
                        if (out_idx < 0 || out_idx >= output.data.size()) {
                            std::printf("\n[CRASH] output.data out of bounds! Index: %d, Size: %lu\n", out_idx, output.data.size());
                            std::printf("Context: batch=%lu, out=%d\n", b, out);
                            std::exit(1);
                        }

                        // if(output_data_ptr){
                        //     output_data_ptr->at(out_idx) = accumulator + bias.data.at(bias_idx);
                        // }
                        output.data.at(out_idx) = accumulator + bias.data.at(bias_idx);
                    }
                }
                // active_task--;
                barrier.count_down();
            });
            
        }

        // while (active_task > 0) {
        //     std::this_thread::yield();
        // }
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