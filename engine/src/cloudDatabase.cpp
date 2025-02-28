#include <iostream>
#include <string>
#include <sstream>
#include <curl/curl.h>

// --------------------------------------------------
// Callback to write data from cURL into an std::string
// --------------------------------------------------
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    // Calculate how many bytes actually received
    size_t totalSize = size * nmemb;

    // Append the data to our std::string
    std::string* buffer = static_cast<std::string*>(userp);
    buffer->append(static_cast<char*>(contents), totalSize);

    return totalSize;
}

std::string encodeSpaces(const std::string& fen) {
    std::string encoded = fen;
    // Replace each ' ' with "%20"
    size_t pos = 0;
    while ((pos = encoded.find(' ', pos)) != std::string::npos) {
        encoded.replace(pos, 1, "%20");
        pos += 3; // skip past "%20"
    }
    return encoded;
}

static std::string removeAllWhitespace(const std::string& str)
{
    std::string result;
    result.reserve(str.size());  // Reserve to avoid repeated allocations

    for (char c : str) {
        // Check if the character is not a whitespace
        if (!std::isspace(static_cast<unsigned char>(c))) {
            result.push_back(c);
        }
    }
    return result;
}


// --------------------------------------------------
// Helper to parse "querybest" response and extract
// the first best move, e.g., "move:e2e4" or "egtb:h2h4".
// --------------------------------------------------
// static std::string parseBestMove(std::string& response)
// {

//     if (!response.empty()) {
//         response.pop_back();  // Erase last char in place
//     }
//     // std::string response = removeAllWhitespace(responseRaw);
//     // Possible error messages returned by the server
//     if (response == "invalid board" ||
//         response == "nobestmove"   ||
//         response == "checkmate"    ||
//         response == "stalemate"    ||
//         response == "unknown" ||
//         response == "nobestmove")
//     {
//         // No valid move can be extracted in these cases
//         std::cout << "Database response: " << response << std::endl;
//         return "nobestmove";
//     }

//     // Typical "querybest" response might look like:
//     //   move:e2e4|search:d2d4|...
//     // or
//     //   egtb:e7e8q
//     // We'll split by '|' and look for prefixes:
//     //   "move:"  -> best move
//     //   "egtb:"  -> EGTB move
//     //   "search:"-> candidate move (fallback)
//     //
//     // We'll return the **first** "move:" or "egtb:" we find.
//     // If none found, we can optionally check for "search:".

//     std::stringstream ss(response);
//     std::string segment;

//     // 1) Try for a normal or EGTB move
//     while (std::getline(ss, segment, '|'))
//     {
//         if (segment.rfind("move:", 0) == 0)
//         {
//             return segment.substr(5); // Remove "move:"
//         }
//         if (segment.rfind("egtb:", 0) == 0)
//         {
//             return segment.substr(5); // Remove "egtb:"
//         }
//     }

//     // 2) If we found no "move:" or "egtb:", we can look for "search:"
//     ss.clear();
//     ss.seekg(0, std::ios::beg); // Reset stream
//     // while (std::getline(ss, segment, '|'))
//     // {
//     //     if (segment.rfind("search:", 0) == 0)
//     //     {
//     //         return segment.substr(7); // Remove "search:"
//     //     }
//     // }

//     // If we can't find any recognized move, return empty
//     return "Error";
// }


static std::string parseBestMove(std::string &response)
{
    // 1) Optionally remove the trailing character if the string isn't empty.
    if (!response.empty()) {
        response.pop_back();  // e.g., remove trailing newline or delimiter
    }

    // 2) Check for known error messages
    if (response == "invalid board" ||
        response == "nobestmove"   ||
        response == "checkmate"    ||
        response == "stalemate"    ||
        response == "unknown")
    {
        std::cout << "Database response: " << response << std::endl;
        return "nobestmove";
    }

    // 3) Look for the first occurrence of "move:"
    //    The format might be:
    //      move:e2e4,score:...,rank:...,note:...,winrate:...|move:d2d4,score:...
    //    or it could be a single line, e.g.:
    //      e2e4,score:1,rank:2,note:! (20-04),winrate:50.08
    //    We want to extract "e2e4" from the first "move:e2e4,..."

    // Step (a): Find "move:" in the string
    const std::string moveTag = "move:";
    size_t movePos = response.find(moveTag);
    if (movePos != std::string::npos)
    {
        // We found "move:". The move is right after that tag.
        size_t start = movePos + moveTag.size(); // position after "move:"

        // Step (b): We read until we hit a comma ',' or a pipe '|' or end of string
        // Example substring: "e2e4,score:1,rank:2..."
        // We'll stop at the first comma or pipe
        size_t end = response.find_first_of(",|", start);
        if (end == std::string::npos) {
            end = response.size(); // no comma/pipe => goes to end
        }

        // Extract the move text
        std::string theMove = response.substr(start, end - start);

        // Trim whitespace just in case
        // (Not strictly needed if the database never returns extra spaces)
        if (!theMove.empty() && (theMove.front() == ' ' || theMove.back() == ' '))
        {
            // Quick trim (optional)
            while(!theMove.empty() && theMove.front() == ' ') theMove.erase(theMove.begin());
            while(!theMove.empty() && theMove.back()  == ' ') theMove.pop_back();
        }

        return theMove;
    }

    // If we reach here, we didn't find "move:"
    // But maybe the response is just "e2e4,score:1..."
    // Let's do a quick fallback search for a leading move if it doesn't have "move:"
    // We'll treat the first token before a comma or pipe as the move
    {
        // We'll look until a comma or pipe or end of string
        size_t end = response.find_first_of(",|");
        std::string potentialMove = (end == std::string::npos)
            ? response  // no comma/pipe => entire string
            : response.substr(0, end);

        // Basic check: is it at least 4 chars to look like "e2e4" etc.
        if (potentialMove.size() >= 4 && potentialMove.size() <= 5) {
            return potentialMove; 
        }
    }

    // If none of the above matched, return Error
    return "Error";
}

// --------------------------------------------------
// Main function to call "querybest" and return best move
// --------------------------------------------------
std::string getBestMoveFromCDB(const std::string& fen)
{
    // --- 1) Build request URL ---
    // Make sure to URL-encode the FEN properly. Here we assume
    // the spaces in FEN are replaced by "%20".
    // Also, we use:
    //   action=querybest
    //   board=[fen]
    //   endgame=0
    //   egtbmetric=dtz
    //   learn=1
    //
    // You can add or remove parameters as needed.
    // std::string baseUrl = "http://www.chessdb.cn/cdb.php?action=querybest&board=";
    std::string baseUrl = "http://www.chessdb.cn/cdb.php?action=queryall&board=";
    // std::string url = baseUrl + "&board=" + fen + "&endgame=0&egtbmetric=dtz&learn=1";

    std::string fenEncoded = encodeSpaces(fen);
// "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201"

std::string url = baseUrl
                + fenEncoded;
                // + "&endgame=0&egtbmetric=dtz&learn=1";

    // --- 2) Initialize cURL and set options ---
    CURL* curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Error: failed to initialize libcurl.\n";
        return "nobestmove";
    }

    // This will store the server's raw response
    std::string response;


    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    // Optional: set a custom user-agent
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "MyChessClient/1.0");

    // --- 3) Perform the request ---
    CURLcode res = curl_easy_perform(curl);

    // --- 4) Clean up cURL ---
    curl_easy_cleanup(curl);

    // If the request failed at the transport level, handle it
    if (res != CURLE_OK)
    {
        std::cerr << "cURL error: " << curl_easy_strerror(res) << "\n";
        return "nobestmove";
    }

    // std::cout << "response raw: " << response << std::endl;

    // --- 5) Parse the response for the best move ---
    return parseBestMove(response);
}

// --------------------------------------------------
// Demo usage
// --------------------------------------------------
// int main()
// {
//     // FEN for the starting position, URL-encoded space as "%20"
//     std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201";

//     // Get best move from CDB
//     std::string bestMove = getBestMoveFromCDB(fen);

//     if (!bestMove.empty())
//     {
//         std::cout << "Best move is: " << bestMove << "\n";
//     }
//     else
//     {
//         std::cout << "No best move found or invalid response.\n";
//     }

//     return 0;
// }
