#include<iostream>
#include "Nexus.hpp"

int main(){
    std::cout << "========================================\n";
    std::cout << "      VALEN HIGH-PERFORMANCE ENGINE     \n";
    std::cout << "========================================\n\n";

    Nexus weights({4,6});

    std::cout << "Allocated flat tensor layout. Shape: (" << weights.shape[0] << ", " << weights.shape[1] << ")\n";
    std::cout << "Total physical float32 buffers tracking on heap: " << weights.data.size() << "\n";
    
    // Stride calculation validation trace
    int target_row = 2;
    int target_col = 5;
    std::cout << "Row-Major Index Mapping for coordinate (" << target_row << ", " << target_col << "): " 
              << weights.get_index(target_row, target_col) << " (Expected: 21)\n\n";
}