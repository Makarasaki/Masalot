#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <string>
#include <vector>

#include <cstdlib> // For rand()
#include <ctime>   // For time()
#include <unordered_map>

#include "Movelist.hpp"
#include "Chess_Test.hpp"
#include "../../training/include/chessnet.h"
#include "../include/data_preparation.h"

ChessPosition createChessPosition(const Board &board,
								  const BoardStatus &status,
								  const uint64_t &epTarget)
{
	// Convert the en passant target into a single-bit bitboard if it's valid
	// uint64_t epBitboard = 0ULL;
	// if (status.HasEPPawn && epTarget.squareIndex >= 0 && epTarget.squareIndex < 64)
	// {
	//     epBitboard = 1ULL << epTarget.squareIndex;
	// }

	ChessPosition position = {
		// White pieces
		rotate180(board.WPawn, status.WhiteMove),
		rotate180(board.WKnight, status.WhiteMove),
		rotate180(board.WBishop, status.WhiteMove),
		rotate180(board.WRook, status.WhiteMove),
		rotate180(board.WQueen, status.WhiteMove),
		rotate180(board.WKing, status.WhiteMove),

		// Black pieces
		rotate180(board.BPawn, status.WhiteMove),
		rotate180(board.BKnight, status.WhiteMove),
		rotate180(board.BBishop, status.WhiteMove),
		rotate180(board.BRook, status.WhiteMove),
		rotate180(board.BQueen, status.WhiteMove),
		rotate180(board.BKing, status.WhiteMove),

		// En passant bitboard
		rotate180(epTarget, status.WhiteMove),

		// White to move, from BoardStatus
		status.WhiteMove,

		// Castling
		// If WhiteMove == true, use white's castling flags for WCastle*
		// and black's for BCastle*; if WhiteMove == false, swap them:
		status.WhiteMove ? status.WCastleL : status.BCastleR, // WCastleR
		status.WhiteMove ? status.WCastleR : status.BCastleL, // WCastleL
		status.WhiteMove ? status.BCastleL : status.WCastleR, // BCastleR
		status.WhiteMove ? status.BCastleR : status.WCastleL  // BCastleL
	};
	return position;
}

class MoveReciever
{
public:
	static inline uint64_t nodes;
	static inline ChessNet model;
	static inline std::unordered_map<uint64_t, float> *evaluations_map;
	static inline std::vector<torch::Tensor> inputs;

	static _ForceInline void Init(Board &brd, uint64_t EPInit, ChessNet trained_model, std::unordered_map<uint64_t, float> &map)
	{
		MoveReciever::nodes = 0;
		MoveReciever::model = trained_model;
		MoveReciever::evaluations_map = &map;
		if (torch::cuda::is_available())
		{
			model->to(torch::kCUDA);
		}
		Movelist::Init(EPInit);
	}

	template <class BoardStatus status>
	static _ForceInline std::uint64_t combineHash(Board &brd, uint64_t EnPassantTarget)
	{
		std::uint64_t h = 0xcbf29ce484222325ULL;
		auto hashCombine = [&](std::uint64_t val)
		{
			// Fowler–Noll–Vo hash function (FNV-1a)
			h ^= val;
			h *= 1099511628211ULL;
		};

		hashCombine(brd.WPawn);
		hashCombine(brd.WKnight);
		hashCombine(brd.WBishop);
		hashCombine(brd.WRook);
		hashCombine(brd.WQueen);
		hashCombine(brd.WKing);
		hashCombine(brd.BPawn);
		hashCombine(brd.BKnight);
		hashCombine(brd.BBishop);
		hashCombine(brd.BRook);
		hashCombine(brd.BQueen);
		hashCombine(brd.BKing);
		hashCombine(EnPassantTarget);

		return h;
	}

	template <class BoardStatus status>
	static _ForceInline ChessData positionToBitboards(Board &brd)
	{
		uint64_t allOnes = ~0ULL;
		ChessData bitboards;
		bitboards.bitboards.resize(13);

		bitboards.bitboards[0] = intToBitboardWhites(brd.WPawn);
		bitboards.bitboards[1] = intToBitboardWhites(brd.WKnight);
		bitboards.bitboards[2] = intToBitboardWhites(brd.WBishop);
		bitboards.bitboards[3] = intToBitboardWhites(brd.WRook);
		bitboards.bitboards[4] = intToBitboardWhites(brd.WQueen);
		bitboards.bitboards[5] = intToBitboardWhites(brd.WKing);

		bitboards.bitboards[6] = intToBitboardBlacks(brd.BPawn);
		bitboards.bitboards[7] = intToBitboardBlacks(brd.BKnight);
		bitboards.bitboards[8] = intToBitboardBlacks(brd.BBishop);
		bitboards.bitboards[9] = intToBitboardBlacks(brd.BRook);
		bitboards.bitboards[10] = intToBitboardBlacks(brd.BQueen);
		bitboards.bitboards[11] = intToBitboardBlacks(brd.BKing);

		bitboards.bitboards[12] = intToBitboard(Movelist::EnPassantTarget);
		return bitboards;
	}

	template <class BoardStatus status>
	static _ForceInline float evaluate(Board &brd)
	{
		ChessPosition position = createChessPosition(brd, status, Movelist::EnPassantTarget);

		torch::Tensor positionINTensor = model->toTensor(position);
		if (torch::cuda::is_available())
		{
			positionINTensor = positionINTensor.to(torch::kCUDA);
		}

		inputs.push_back(positionINTensor);

		// Perform the forward pass with the model
		// torch::Tensor output = model->forward(positionINTensor);

		// float eval_value = output.item<float>();
		// (*evaluations_map)[key] = eval_value;

		// std::cout << "Evaluation: " << eval_value << std::endl;
		// return eval_value;
		return 0;
	}

	template <class BoardStatus status>
	static _ForceInline float runBatch()
	{
		torch::Tensor batch_inputs = torch::stack(inputs);
		torch::Tensor output = model->forward(batch_inputs);
		inputs.clear();

		if constexpr (status.WhiteMove)
		{
			// For White's move, return the highest evaluation
			float max_val = output.max().item<float>();
			// std::cout << "WhiteMove: Returning max value = " << max_val << std::endl;
			return max_val;
		}
		else
		{
			// For Black's move, return the lowest evaluation
			float min_val = output.min().item<float>();
			// std::cout << "BlackMove: Returning min value = " << min_val << std::endl;
			return min_val;
		}
	}

	template <class BoardStatus status>
	static _ForceInline float PerfT0(Board &brd)
	{
		// std::cout << "Status PERFT0" << std::endl;
		nodes++;
		// return 0;
		float eval = evaluate<status>(brd);
		// float eval = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / 2.0f) - 1.0f;//std::cout << "eval: " << eval << std::endl;
		return eval;
	}

	template <class BoardStatus status>
	static _ForceInline void PerfT1(Board &brd)
	{
		// std::cout << "PerfT1" << std::endl;
		nodes += Movelist::count<status>(brd);
	}

	template <class BoardStatus status, int depth>
	static _ForceInline float PerfT(Board &brd, float alpha, float beta)
	{
		// Movelist::EnumerateMoves<status, MoveReciever, depth>(brd);
		if constexpr (depth == 0)
			return PerfT0<status>(brd);
		else
			// Tutaj movereceiver jest przekazywany jako Callback_Move
			return Movelist::EnumerateMoves<status, MoveReciever, depth>(brd, alpha, beta);
	}

#define ENABLEDBG 0
#define ENABLEPRINT 0
#define IFDBG if constexpr (ENABLEDBG)
#define IFPRN if constexpr (ENABLEPRINT)

	// template<class BoardStatus status, int depth>
	// static float SomeMate(const Board& brd, float alpha, float beta)
	//{
	//	return PerfT0<status>(brd);
	// }

	template <class BoardStatus status, int depth>
	static float Kingmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::King, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Kingmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		return PerfT<status.KingMove(), depth - 1>(next, alpha, beta);
	}

	template <class BoardStatus status, int depth>
	static float KingCastle(const Board &brd, uint64_t kingswitch, uint64_t rookswitch, float alpha, float beta)
	{
		Board next = Board::MoveCastle<status.WhiteMove>(brd, kingswitch, rookswitch);
		IFPRN std::cout << "KingCastle:\n"
						<< _map(kingswitch, rookswitch, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, false);
		return PerfT<status.KingMove(), depth - 1>(next, alpha, beta);
	}

	template <class BoardStatus status, int depth>
	static void PawnCheck(map eking, uint64_t to)
	{
		constexpr bool white = status.WhiteMove;
		map pl = Pawn_AttackLeft<white>(to & Pawns_NotLeft());
		map pr = Pawn_AttackRight<white>(to & Pawns_NotRight());

		if (eking & (pl | pr))
			Movestack::Check_Status[depth - 1] = to;
	}

	template <class BoardStatus status, int depth>
	static void KnightCheck(map eking, uint64_t to)
	{
		constexpr bool white = status.WhiteMove;

		if (Lookup::Knight(SquareOf(eking)) & to)
			Movestack::Check_Status[depth - 1] = to;
	}

	template <class BoardStatus status, int depth>
	static float Pawnmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, false>(brd, from, to);
		IFPRN std::cout << "Pawnmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);

		// globalMoveList.emplace_back(next, status.SilentMove(), Movelist::EnPassantTarget);

		float eval = PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Pawnatk(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, true>(brd, from, to);
		IFPRN std::cout << "Pawntake:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float PawnEnpassantTake(const Board &brd, uint64_t from, uint64_t enemy, uint64_t to, float alpha, float beta)
	{
		Board next = Board::MoveEP<status.WhiteMove>(brd, from, enemy, to);
		IFPRN std::cout << "PawnEnpassantTake:\n"
						<< _map(from | enemy, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, true);
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Pawnpush(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, false>(brd, from, to);
		IFPRN std::cout << "Pawnpush:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));

		Movelist::EnPassantTarget = to;
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<status.PawnPush(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	// REQUIRES MULTIPLE EVALUATIONS TO BE RETURNED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	template <class BoardStatus status, int depth>
	static float Pawnpromote(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next1 = Board::MovePromote<BoardPiece::Queen, status.WhiteMove>(brd, from, to);
		IFPRN std::cout << "Pawnpromote:\n"
						<< _map(from, to, brd, next1) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next1, to & Enemy<status.WhiteMove>(brd));
		float eval1 = PerfT<status.SilentMove(), depth - 1>(next1, alpha, beta);

		Board next2 = Board::MovePromote<BoardPiece::Knight, status.WhiteMove>(brd, from, to);
		KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval2 = PerfT<status.SilentMove(), depth - 1>(next2, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;

		Board next3 = Board::MovePromote<BoardPiece::Bishop, status.WhiteMove>(brd, from, to);
		float eval3 = PerfT<status.SilentMove(), depth - 1>(next3, alpha, beta);
		Board next4 = Board::MovePromote<BoardPiece::Rook, status.WhiteMove>(brd, from, to);
		float eval4 = PerfT<status.SilentMove(), depth - 1>(next4, alpha, beta);
		if constexpr (status.WhiteMove)
		{
			// For White's move, return the highest evaluation
			return std::max({eval1, eval2, eval3, eval4});
		}
		else
		{
			// For Black's move, return the lowest evaluation
			return std::min({eval1, eval2, eval3, eval4});
		}
	}

	template <class BoardStatus status, int depth>
	static float Knightmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Knight, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Knightmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Bishopmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Bishop, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Bishopmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		return PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
	}

	template <class BoardStatus status, int depth>
	static float Rookmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Rook, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Rookmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		if constexpr (status.CanCastle())
		{
			if (status.IsLeftRook(from))
				return PerfT<status.RookMove_Left(), depth - 1>(next, alpha, beta);
			else if (status.IsRightRook(from))
				return PerfT<status.RookMove_Right(), depth - 1>(next, alpha, beta);
			else
				return PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		else
			return PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
	}

	template <class BoardStatus status, int depth>
	static float Queenmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Queen, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Queenmove:\n"
						<< _map(from, to, brd, next) << "\n";
		// IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		return PerfT<status.SilentMove(), depth - 1>(next, alpha, beta);
	}
};

template <class BoardStatus status>
static float PerfT(std::string_view def, Board &brd, int depth, float alpha, float beta, ChessNet model, std::unordered_map<uint64_t, float> &evaluations_map)
{
	MoveReciever::Init(brd, FEN::FenEnpassant(def), model, evaluations_map);

	switch (depth)
	{
	case 0:
		Movelist::InitStack<status, 0>(brd);
		return MoveReciever::PerfT0<status>(brd);
	case 1:
		Movelist::InitStack<status, 1>(brd);
		return MoveReciever::PerfT<status, 1>(brd, alpha, beta); // Keep this as T1
	case 2:
		Movelist::InitStack<status, 2>(brd);
		return MoveReciever::PerfT<status, 2>(brd, alpha, beta);
	case 3:
		Movelist::InitStack<status, 3>(brd);
		return MoveReciever::PerfT<status, 3>(brd, alpha, beta);
	case 4:
		Movelist::InitStack<status, 4>(brd);
		return MoveReciever::PerfT<status, 4>(brd, alpha, beta);
	case 5:
		Movelist::InitStack<status, 5>(brd);
		return MoveReciever::PerfT<status, 5>(brd, alpha, beta);
	case 6:
		Movelist::InitStack<status, 6>(brd);
		return MoveReciever::PerfT<status, 6>(brd, alpha, beta);
	case 7:
		Movelist::InitStack<status, 7>(brd);
		return MoveReciever::PerfT<status, 7>(brd, alpha, beta);
	case 8:
		Movelist::InitStack<status, 8>(brd);
		return MoveReciever::PerfT<status, 8>(brd, alpha, beta);
	case 9:
		Movelist::InitStack<status, 9>(brd);
		return MoveReciever::PerfT<status, 9>(brd, alpha, beta);
	case 10:
		Movelist::InitStack<status, 10>(brd);
		return MoveReciever::PerfT<status, 10>(brd, alpha, beta);
	case 11:
		Movelist::InitStack<status, 11>(brd);
		return MoveReciever::PerfT<status, 11>(brd, alpha, beta);
	case 12:
		Movelist::InitStack<status, 12>(brd);
		return MoveReciever::PerfT<status, 12>(brd, alpha, beta);
	case 13:
		Movelist::InitStack<status, 13>(brd);
		return MoveReciever::PerfT<status, 13>(brd, alpha, beta);
	case 14:
		Movelist::InitStack<status, 14>(brd);
		return MoveReciever::PerfT<status, 14>(brd, alpha, beta);
	case 15:
		Movelist::InitStack<status, 15>(brd);
		return MoveReciever::PerfT<status, 15>(brd, alpha, beta);
	case 16:
		Movelist::InitStack<status, 16>(brd);
		return MoveReciever::PerfT<status, 16>(brd, alpha, beta);
	case 17:
		Movelist::InitStack<status, 17>(brd);
		return MoveReciever::PerfT<status, 17>(brd, alpha, beta);
	case 18:
		Movelist::InitStack<status, 18>(brd);
		return MoveReciever::PerfT<status, 18>(brd, alpha, beta);
	default:
		std::cout << "Depth not impl yet" << std::endl;
		return 2137;
	}
}
PositionToTemplate(PerfT);

const auto _keep0 = _map(0);
const auto _keep1 = _map(0, 0);
const auto _keep2 = _map(0, 0, 0);
const auto _keep4 = _map(0, 0, Board::Default(), Board::Default());

// int main(int argc, char** argv)
// {
// 	std::random_device rd;
// 	std::mt19937_64 eng(rd());
// 	std::uniform_int_distribution<unsigned long long> distr;
// 	srand(static_cast<unsigned int>(time(0)));

// 	//std::string_view dbg = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

// 	std::string_view def = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// 	std::string_view kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
// 	//std::string_view pintest = "6Q1/8/4k3/8/4r3/1K6/4R3/8 b - - 0 1";
// 	//std::string_view def = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";
// 	std::string_view midgame = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
// 	std::string_view endgame = "5nk1/pp3pp1/2p4p/q7/2PPB2P/P5P1/1P5K/3Q4 w - - 1 28";

// 	std::string_view stalemate = "7k/5Q2/8/8/8/8/8/K7 b - - 0 1";
// 	std::string_view checkmate = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";

// 	std::string_view double_checkmate = "6rk/5N2/5K2/8/8/8/8/7Q b - - 0 1";
// 	std::string_view almost_stalemate = "7k/4Q3/5K2/8/8/8/8/8 w - - 0 1";
// 	std::string_view onepawn = "7k/6p1/8/8/8/8/8/K7 w - - 0 1";
// 	//55.8

// 	std::cout << "Start" << std::endl;
// 	uint64_t depth = 8;
// 	float alpha = std::numeric_limits<float>::lowest();
// 	float beta = std::numeric_limits<float>::max();
// 	auto start = std::chrono::steady_clock::now();

// 	float evaluation = _PerfT(endgame, depth, alpha, beta);
// 	auto end = std::chrono::steady_clock::now();
// 	long long delta = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
// 	std::cout << "Position evaluation: " << evaluation << std::endl;
// 	std::cout << "Czas: " << MoveReciever::nodes << " " << delta / 1000 << "ms " << MoveReciever::nodes * 1.0 / delta << " MNodes/s\n";
// 	std::cout << "Koniec mojego testu" << std::endl;
// 	std::cout << "nodes: " << MoveReciever::nodes << std::endl;
// 	return 0;

// }
