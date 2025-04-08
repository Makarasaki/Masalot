#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // std::min_element
#include <stdexcept>

// Include nlohmann/json (https://github.com/nlohmann/json)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Include curl headers
#include <curl/curl.h>

/**
 * Helper callback for libcurl to write data into a std::string.
 */
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    auto totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

/**
 * Perform a GET request to the given URL using libcurl and return the response body as a std::string.
 * Throws std::runtime_error on error.
 */
std::string httpGet(const std::string& url)
{
    CURL* curl = curl_easy_init();
    if(!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // For HTTPS; ignore in this example if site uses plain HTTP. 
    // If you need HTTPS with certificate verification, consider:
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    CURLcode res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error(std::string("curl_easy_perform() failed: ") + curl_easy_strerror(res));
    }

    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);

    if(response_code < 200 || response_code >= 300) {
        throw std::runtime_error("HTTP request failed with status code " + std::to_string(response_code));
    }

    return response;
}

/**
 * Given a FEN string, query the Lichess tablebase, parse JSON, and return a best move in UCI format.
 *
 * Selection strategy (customize as desired):
 *   1. Filter moves with "category" = "win".
 *   2. Among those, choose the one with the smallest "dtm".
 *   3. If no "win" moves exist, pick "draw" if it exists,
 *      otherwise pick any move if "loss" is all that's left.
 *
 * If there is no valid move in the JSON, returns an empty string or throws.
 */
std::string getBestMoveFromTablebase(const std::string& fen)
{
    // Construct the tablebase URL
    // e.g., "http://tablebase.lichess.ovh/standard?fen=..."
    // URL-encode the fen if it has spaces, plus signs, etc. for safety.
    // For brevity, this example does not URL-encode. If your FEN is guaranteed
    // to have no spaces or special characters, you may skip. Otherwise, encode properly.
    std::string baseUrl = "http://tablebase.lichess.ovh/standard?fen=";
    std::string url = baseUrl + fen;

    // Get JSON response
    std::string response = httpGet(url);

    // Parse JSON
    auto j = json::parse(response);

    if(!j.contains("moves") || !j["moves"].is_array()) {
        throw std::runtime_error("JSON response does not contain 'moves' array");
    }

    // Extract all moves
    auto movesArray = j["moves"];

    // We can store them in a struct for easy handling
    struct MoveInfo {
        std::string uci;
        std::string category;
        int dtm;
    };

    std::vector<MoveInfo> allMoves;
    allMoves.reserve(movesArray.size());

    for(const auto& moveJson : movesArray) {
        MoveInfo mi;
        mi.uci      = moveJson.value("uci", "");
        mi.category = moveJson.value("category", "unknown");
        // dtm might be null in some cases, so we handle that carefully
        if(moveJson.contains("dtm") && !moveJson["dtm"].is_null()) {
            mi.dtm = moveJson["dtm"].get<int>();
        } else {
            mi.dtm = 999999; // or some large number if dtm is not available
        }

        // Only store if it has a UCI string
        if(!mi.uci.empty()) {
            allMoves.push_back(mi);
        }
    }

    if(allMoves.empty()) {
        // No moves found
        return "";
    }

    // Filter by category: look for "win" first
    std::vector<MoveInfo> winMoves;
    for (auto &m: allMoves) {
        if(m.category == "win") {
            winMoves.push_back(m);
        }
    }

    if(!winMoves.empty()) {
        // Choose move with smallest dtm
        auto best = std::min_element(winMoves.begin(), winMoves.end(), 
            [](const MoveInfo& a, const MoveInfo& b){
                return a.dtm < b.dtm;
            }
        );
        return best->uci;
    }

    // If no "win", try "draw"
    std::vector<MoveInfo> drawMoves;
    for(auto &m: allMoves) {
        if(m.category == "draw") {
            drawMoves.push_back(m);
        }
    }

    if(!drawMoves.empty()) {
        // Could pick any, or pick by dtm
        return drawMoves.front().uci;
    }

    // Otherwise, pick a "loss" or any leftover
    // (In practice, you'd pick whichever fits your engine's logic.)
    return allMoves.front().uci;
}

