#ifndef DATA_PREPARATION_H
#define DATA_PREPARATION_H

#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <string>


struct ChessData {
    std::vector<std::vector<std::vector<int>>> bitboards;  // 3D vector: 14 bitboards, each 8x8
};

ChessData fenToBitboards(const std::string& pos);

torch::Tensor bitboardsToTensor(const std::vector<std::vector<std::vector<int>>>& bitboards);


#endif  // DATA_PREPARATION_H