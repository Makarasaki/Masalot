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

const int PORT = 12346;
const int BUFFER_SIZE = 1024;

void handle_connection(int client_socket) {
    char buffer[BUFFER_SIZE];

    // Load the model
    ChessNet model;
    torch::serialize::InputArchive input_archive;
    try {
        input_archive.load_from("../../training/NN_weights/model_last.pt");
        model.load(input_archive);  // Load the weights into the model
        model.eval();
        // Check if CUDA is available
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA);
            std::cout << "Using CUDA" << std::endl;
        } else {
            // If not available, keep on CPU
            model.to(torch::kCPU);
        }
        std::cout << "Model weights loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model weights: " << e.what() << std::endl;
        close(client_socket); // Close the socket in case of an error
        return;
    }

    // Loop to handle multiple FEN strings in the same connection
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
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string best_move_fen = search_best_move(model, received_fen, 4);  // Depth set to 2 for example
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        std::cout << "Time taken to find best move: " << duration.count() << " seconds." << std::endl;
        std::cout << "Best move: " << best_move_fen << std::endl;

        // Send the best move FEN back to the client.
        if (write(client_socket, best_move_fen.c_str(), best_move_fen.length()) < 0) {
            std::cerr << "Failed to write to socket" << std::endl;
            close(client_socket);  // Handle error and close connection
            return;
        }

        // Loop back to read the next move from the client
        std::cout << "Waiting for the next move..." << std::endl;
    }
}

int main() {
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << std::endl;
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

    // Keep running to accept connections
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
}

