#include <iostream>
#include <vector>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <limits>
#include <memory>
#include "../include/evaluate.h"
// #include "../include/chessnet.h"

// Constants
const int PORT = 12346;
const int BUFFER_SIZE = 1024;

// Placeholder struct for a chess position.
struct Position {
    std::string fen; // FEN representation of the position
    // You can add other data members (bitboards, etc.)
};


// Alpha-beta pruning algorithm.
// int alpha_beta(Position pos, int depth, int alpha, int beta, bool isWhite) {
//     if (depth == 0) {
//         return evaluate(pos);
//     }

//     std::vector<Position> positions = generate_positions(pos);
//     if (positions.empty()) {
//         return evaluate(pos);  // Handle checkmate or stalemate, depending on the side to move.
//     }

//     if (isWhite) {
//         int max_eval = std::numeric_limits<int>::min();
//         for (const auto& new_pos : positions) {
//             int eval = alpha_beta(new_pos, depth - 1, alpha, beta, false);
//             max_eval = std::max(max_eval, eval);
//             alpha = std::max(alpha, eval);
//             if (beta <= alpha) break;  // Beta cutoff
//         }
//         return max_eval;
//     } else {
//         int min_eval = std::numeric_limits<int>::max();
//         for (const auto& new_pos : positions) {
//             int eval = alpha_beta(new_pos, depth - 1, alpha, beta, true);
//             min_eval = std::min(min_eval, eval);
//             beta = std::min(beta, eval);
//             if (beta <= alpha) break;  // Alpha cutoff
//         }
//         return min_eval;
//     }
// }


// Convert a FEN string to a Position (dummy implementation).
Position parse_fen(const std::string& fen) {
    Position pos;
    pos.fen = fen;
    // Add FEN parsing logic here.
    return pos;
}

// Convert a Position to a FEN string (dummy implementation).
std::string to_fen(const Position& pos) {
    // Convert the Position back to a FEN string.
    return pos.fen;
}

// void handle_connection(int client_socket) {
//     char buffer[BUFFER_SIZE];

//     ChessNet model;
//     torch::serialize::InputArchive input_archive;
//     try {
//         input_archive.load_from("/mnt/c/MAKS_STUDIA/MGR/test/training3/NN_weights/test_model.pt");
//         model.load(input_archive);  // Load the weights into the model
//         std::cout << "Model weights loaded successfully!" << std::endl;
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading model weights: " << e.what() << std::endl;
//     }

//     while (true) {
//         memset(buffer, 0, BUFFER_SIZE);

//         // Read the incoming FEN string from the client.
//         int bytes_read = read(client_socket, buffer, BUFFER_SIZE);
//         if (bytes_read < 0) {
//             std::cerr << "Failed to read from socket" << std::endl;
//             close(client_socket);
//             return;
//         }

//         std::string received_fen(buffer);

//         // Check if the received message is "end".
//         if (received_fen == "end") {
//             std::cout << "Received 'end' message, closing connection." << std::endl;
//             close(client_socket);
//             break;
//         }

//         std::cout << "Received FEN: " << received_fen << std::endl;


//         std::string best_move_fen = search_best_move(model, received_fen, 2);

//         std::cout << "dupa" << std::endl;
//         std::cout << best_move_fen << std::endl;


//         // Send the best move FEN back to the client.
//         write(client_socket, best_move_fen.c_str(), best_move_fen.length());
//     }
// }

void handle_connection(int client_socket) {
    char buffer[BUFFER_SIZE];

    // Load the model
    ChessNet model;
    torch::serialize::InputArchive input_archive;
    try {
        input_archive.load_from("/mnt/c/MAKS_STUDIA/MGR/test/training3/NN_weights/test_model.pt");
        model.load(input_archive);  // Load the weights into the model
        std::cout << "Model weights loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model weights: " << e.what() << std::endl;
        close(client_socket); // Close the socket in case of an error
        return;
    }

    // Infinite loop to handle client requests
    while (true) {
        memset(buffer, 0, BUFFER_SIZE);

        // Read the incoming FEN string from the client.
        int bytes_read = read(client_socket, buffer, BUFFER_SIZE - 1);
        if (bytes_read < 0) {
            std::cerr << "Failed to read from socket" << std::endl;
            close(client_socket);  // Handle error and close connection
            return;
        }

        if (bytes_read == 0) {
            std::cout << "Client disconnected." << std::endl;
            close(client_socket);  // Handle graceful client disconnection
            return;
        }

        std::string received_fen(buffer, bytes_read);  // Construct string from buffer

        // Check if the received message is "end".
        if (received_fen == "end") {
            std::cout << "Received 'end' message, closing connection." << std::endl;
            close(client_socket);  // Handle termination request
            break;
        }

        std::cout << "Received FEN: " << received_fen << std::endl;

        // Get the best move FEN from the model
        std::string best_move_fen = search_best_move(model, received_fen, 1);  // Depth set to 2 for example

        std::cout << "Best move: " << best_move_fen << std::endl;

        // Send the best move FEN back to the client.
        if (write(client_socket, best_move_fen.c_str(), best_move_fen.length()) < 0) {
            std::cerr << "Failed to write to socket" << std::endl;
            close(client_socket);  // Handle error and close connection
            return;
        }
    }
}


int main() {
    int server_fd, client_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation error" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Define the server address
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind the socket to the port
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_fd, 3) < 0) {
        std::cerr << "Listen failed" << std::endl;
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    while (true) {
        // Accept a connection from a client
        if ((client_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            std::cerr << "Accept failed" << std::endl;
            close(server_fd);
            exit(EXIT_FAILURE);
        }

        // Handle the connection (read FEN, find best move, return FEN)
        handle_connection(client_socket);
    }


    // // std::string position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // std::string position = "8/8/4K3/8/8/8/2q2r2/k7 w - - 0 1";

    // // generate_positions(position, false);

    // ChessNet model;
    // torch::serialize::InputArchive input_archive;
    // try {
    //     input_archive.load_from("/mnt/c/MAKS_STUDIA/MGR/test/training3/NN_weights/test_model.pt");
    //     model.load(input_archive);  // Load the weights into the model
    //     std::cout << "Model weights loaded successfully!" << std::endl;
    // } catch (const c10::Error& e) {
    //     std::cerr << "Error loading model weights: " << e.what() << std::endl;
    //     return -1;
    // }

    // float eval = alpha_beta(model, position, 2, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), true);

    // std::cout << eval << std::endl;


    // std::cout << search_best_move(model, position, 2) << std::endl;
    return 0;
}
