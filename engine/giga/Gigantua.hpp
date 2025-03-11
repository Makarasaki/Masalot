// https://github.com/Gigantua/Gigantua
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
#include "../include/zorbist.hpp"

ChessPosition createChessPosition(const Board &board,
								  const BoardStatus &status,
								  const uint64_t &epTarget)
{

	ChessPosition position = [&]()
	{
		if (status.WhiteMove)
		{
			return ChessPosition(
				// White pieces
				board.WPawn,
				board.WKnight,
				board.WBishop,
				board.WRook,
				board.WQueen,
				board.WKing,

				// Black pieces
				board.BPawn,
				board.BKnight,
				board.BBishop,
				board.BRook,
				board.BQueen,
				board.BKing,

				// En passant bitboard
				epTarget,

				// White to move, from BoardStatus
				status.WhiteMove,

				// Castling
				// If WhiteMove == true, use white's castling flags for WCastle*
				// and black's for BCastle*; if WhiteMove == false, swap them:
				status.WCastleL,
				status.WCastleR,
				status.BCastleL,
				status.BCastleR);
		}
		else
		{
			return ChessPosition(
				// Black pieces
				flipVertical(board.BPawn, status.WhiteMove),
				flipVertical(board.BKnight, status.WhiteMove),
				flipVertical(board.BBishop, status.WhiteMove),
				flipVertical(board.BRook, status.WhiteMove),
				flipVertical(board.BQueen, status.WhiteMove),
				flipVertical(board.BKing, status.WhiteMove),

				// White pieces
				flipVertical(board.WPawn, status.WhiteMove),
				flipVertical(board.WKnight, status.WhiteMove),
				flipVertical(board.WBishop, status.WhiteMove),
				flipVertical(board.WRook, status.WhiteMove),
				flipVertical(board.WQueen, status.WhiteMove),
				flipVertical(board.WKing, status.WhiteMove),

				// En passant bitboard
				flipVertical(epTarget, status.WhiteMove),

				// White to move, from BoardStatus
				status.WhiteMove,

				// Castling
				// If WhiteMove == true, use white's castling flags for WCastle*
				// and black's for BCastle*; if WhiteMove == false, swap them:
				status.BCastleL,
				status.BCastleR,
				status.WCastleL,
				status.WCastleR);
		}
	}();

	return position;
}

class MoveReceiver
{
public:
	static inline uint64_t nodes;
	static inline ChessNet model;
	static inline std::unordered_map<uint64_t, float> *evaluations_map;
	static inline std::vector<torch::Tensor> inputs;

	static _ForceInline void Init(Board &brd, uint64_t EPInit, ChessNet trained_model, std::unordered_map<uint64_t, float> &map)
	{
		MoveReceiver::nodes = 0;
		MoveReceiver::model = trained_model;
		MoveReceiver::evaluations_map = &map;
		if (torch::cuda::is_available())
		{
			model->to(torch::kCUDA);
			// model->to(torch::kCPU);
		}
		Movelist::Init(EPInit);
		initZobristKeys();
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
	static _ForceInline float evaluate(Board &brd)
	{
		// 1. Generate a unique key for the current position
		uint64_t key = computeZobristHash(brd, status, Movelist::EnPassantTarget);

		// 2. Check if we have a cached evaluation for this position
		auto it = evaluations_map->find(key);
		if (it != evaluations_map->end())
		{
			return it->second;
		}

		ChessPosition position = createChessPosition(brd, status, Movelist::EnPassantTarget);

		torch::Tensor positionINTensor = model->toTensor(position);

		positionINTensor = positionINTensor.unsqueeze(0);
		if (torch::cuda::is_available())
		{
			positionINTensor = positionINTensor.to(torch::kCUDA);
		}

		// Perform the forward pass with the model
		torch::Tensor output = model->forward(positionINTensor);

		float eval_value = output.item<float>();

		if (!status.WhiteMove)
		{
			eval_value *= -1;
		}
		(*evaluations_map)[key] = eval_value;

		return eval_value;
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
			return max_val;
		}
		else
		{
			// For Black's move, return the lowest evaluation
			float min_val = output.min().item<float>();
			return min_val;
		}
	}

	template <class BoardStatus status>
	static _ForceInline float PerfT0(Board &brd)
	{
		nodes++;
		float eval = evaluate<status>(brd);
		return eval;
	}

	template <class BoardStatus status>
	static _ForceInline void PerfT1(Board &brd)
	{
		nodes += Movelist::count<status>(brd);
	}

	template <bool IsAttacking, class BoardStatus status, int depth>
	static _ForceInline float PerfT(Board &brd, float alpha, float beta)
	{
		static_assert(depth >= 0, "No negative depth allowed.");
		if constexpr (depth == 0)
		{
			return PerfT0<status>(brd);
		}
		else if constexpr (!IsAttacking && depth == 0)
		{
			return PerfT0<status>(brd);
		}
		else
		{
			return Movelist::EnumerateMoves<status, MoveReceiver, depth>(brd, alpha, beta);
		}
	}

#define ENABLEDBG 1
#define ENABLEPRINT 0
#define IFDBG if constexpr (ENABLEDBG)
#define IFPRN if constexpr (ENABLEPRINT)

	template <class BoardStatus status, int depth>
	static float Kingmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::King, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Kingmove:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		bool isAttacking = (brd.Occ & to) != 0;
		if (isAttacking)
		{
			return PerfT<true, status.KingMove(), depth - 1>(next, alpha, beta);
		}
		else
		{
			return PerfT<false, status.KingMove(), depth - 1>(next, alpha, beta);
		}
	}

	template <class BoardStatus status, int depth>
	static float KingCastle(const Board &brd, uint64_t kingswitch, uint64_t rookswitch, float alpha, float beta)
	{
		Board next = Board::MoveCastle<status.WhiteMove>(brd, kingswitch, rookswitch);
		IFPRN std::cout << "KingCastle:\n"
						<< _map(kingswitch, rookswitch, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, false);
		return PerfT<false, status.KingMove(), depth - 1>(next, alpha, beta);
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
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);

		float eval = PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);

		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Pawnatk(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, true>(brd, from, to);
		IFPRN std::cout << "Pawntake:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		bool isAttacking = (brd.Occ & to) != 0;
		float eval;
		if (isAttacking)
		{
			eval = PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		else
		{
			eval = PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float PawnEnpassantTake(const Board &brd, uint64_t from, uint64_t enemy, uint64_t to, float alpha, float beta)
	{
		Board next = Board::MoveEP<status.WhiteMove>(brd, from, enemy, to);
		IFPRN std::cout << "PawnEnpassantTake:\n"
						<< _map(from | enemy, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, true);
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Pawnpush(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, false>(brd, from, to);
		IFPRN std::cout << "Pawnpush:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));

		Movelist::EnPassantTarget = to;
		PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval = PerfT<false, status.PawnPush(), depth - 1>(next, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Pawnpromote(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next1 = Board::MovePromote<BoardPiece::Queen, status.WhiteMove>(brd, from, to);
		IFPRN std::cout << "Pawnpromote:\n"
						<< _map(from, to, brd, next1) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next1, to & Enemy<status.WhiteMove>(brd));
		float eval1 = PerfT<false, status.SilentMove(), depth - 1>(next1, alpha, beta);

		Board next2 = Board::MovePromote<BoardPiece::Knight, status.WhiteMove>(brd, from, to);
		KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
		float eval2 = PerfT<false, status.SilentMove(), depth - 1>(next2, alpha, beta);
		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;

		Board next3 = Board::MovePromote<BoardPiece::Bishop, status.WhiteMove>(brd, from, to);
		float eval3 = PerfT<false, status.SilentMove(), depth - 1>(next3, alpha, beta);
		Board next4 = Board::MovePromote<BoardPiece::Rook, status.WhiteMove>(brd, from, to);
		float eval4 = PerfT<false, status.SilentMove(), depth - 1>(next4, alpha, beta);
		if constexpr (status.WhiteMove)
		{
			return std::max({eval1, eval2, eval3, eval4});
		}
		else
		{
			return std::min({eval1, eval2, eval3, eval4});
		}
	}

	template <class BoardStatus status, int depth>
	static float Knightmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Knight, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Knightmove:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);

		bool isAttacking = (brd.Occ & to) != 0;
		float eval;
		if (isAttacking)
		{
			eval = PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		else
		{
			eval = PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
		}

		Movestack::Check_Status[depth - 1] = 0xffffffffffffffffull;
		return eval;
	}

	template <class BoardStatus status, int depth>
	static float Bishopmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Bishop, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Bishopmove:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		bool isAttacking = (brd.Occ & to) != 0;
		if (isAttacking)
		{
			return PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		else
		{
			return PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
	}

	template <class BoardStatus status, int depth>
	static float Rookmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Rook, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Rookmove:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		bool isAttacking = (brd.Occ & to) != 0;
		if constexpr (status.CanCastle())
		{
			if (status.IsLeftRook(from))
				return PerfT<false, status.RookMove_Left(), depth - 1>(next, alpha, beta);
			else if (status.IsRightRook(from))
				return PerfT<false, status.RookMove_Right(), depth - 1>(next, alpha, beta);
			else
			{
				if (isAttacking)
				{
					return PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
				}
				else
				{
					return PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
				}
			}
		}
		else
		{
			if (isAttacking)
			{
				return PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
			}
			else
			{
				return PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
			}
		}
	}

	template <class BoardStatus status, int depth>
	static float Queenmove(const Board &brd, uint64_t from, uint64_t to, float alpha, float beta)
	{
		Board next = Board::Move<BoardPiece::Queen, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
		IFPRN std::cout << "Queenmove:\n"
						<< _map(from, to, brd, next) << "\n";
		IFDBG Board::AssertBoardMove<status.WhiteMove>(brd, next, to & Enemy<status.WhiteMove>(brd));
		bool isAttacking = (brd.Occ & to) != 0;
		if (isAttacking)
		{
			return PerfT<true, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
		else
		{
			return PerfT<false, status.SilentMove(), depth - 1>(next, alpha, beta);
		}
	}
};

template <class BoardStatus status>
static float PerfT(std::string_view def, Board &brd, int depth, float alpha, float beta, ChessNet &model, std::unordered_map<uint64_t, float> &evaluations_map)
{
	MoveReceiver::Init(brd, FEN::FenEnpassant(def), model, evaluations_map);

	switch (depth)
	{
	case 0:
		Movelist::InitStack<status, 0>(brd);
		return MoveReceiver::PerfT0<status>(brd);
	case 1:
		Movelist::InitStack<status, 1>(brd);
		return MoveReceiver::PerfT<false, status, 1>(brd, alpha, beta); // Keep this as T1
	case 2:
		Movelist::InitStack<status, 2>(brd);
		return MoveReceiver::PerfT<false, status, 2>(brd, alpha, beta);
	case 3:
		Movelist::InitStack<status, 3>(brd);
		return MoveReceiver::PerfT<false, status, 3>(brd, alpha, beta);
	case 4:
		Movelist::InitStack<status, 4>(brd);
		return MoveReceiver::PerfT<false, status, 4>(brd, alpha, beta);
	case 5:
		Movelist::InitStack<status, 5>(brd);
		return MoveReceiver::PerfT<false, status, 5>(brd, alpha, beta);
	case 6:
		Movelist::InitStack<status, 6>(brd);
		return MoveReceiver::PerfT<false, status, 6>(brd, alpha, beta);
	case 7:
		Movelist::InitStack<status, 7>(brd);
		return MoveReceiver::PerfT<false, status, 7>(brd, alpha, beta);
	case 8:
		Movelist::InitStack<status, 8>(brd);
		return MoveReceiver::PerfT<false, status, 8>(brd, alpha, beta);
	// case 9:
	// 	Movelist::InitStack<status, 9>(brd);
	// 	return MoveReceiver::PerfT<false, status, 9>(brd, alpha, beta);
	// case 10:
	// 	Movelist::InitStack<status, 10>(brd);
	// 	return MoveReceiver::PerfT<false, status, 10>(brd, alpha, beta);
	// case 11:
	// 	Movelist::InitStack<status, 11>(brd);
	// 	return MoveReceiver::PerfT<false, status, 11>(brd, alpha, beta);
	// case 12:
	// 	Movelist::InitStack<status, 12>(brd);
	// 	return MoveReceiver::PerfT<false, status, 12>(brd, alpha, beta);
	// case 13:
	// 	Movelist::InitStack<status, 13>(brd);
	// 	return MoveReceiver::PerfT<false, status, 13>(brd, alpha, beta);
	// case 14:
	// 	Movelist::InitStack<status, 14>(brd);
	// 	return MoveReceiver::PerfT<false, status, 14>(brd, alpha, beta);
	// case 15:
	// 	Movelist::InitStack<status, 15>(brd);
	// 	return MoveReceiver::PerfT<false, status, 15>(brd, alpha, beta);
	// case 16:
	// 	Movelist::InitStack<status, 16>(brd);
	// 	return MoveReceiver::PerfT<false, status, 16>(brd, alpha, beta);
	// case 17:
	// 	Movelist::InitStack<status, 17>(brd);
	// 	return MoveReceiver::PerfT<false, status, 17>(brd, alpha, beta);
	// case 18:
	// 	Movelist::InitStack<status, 18>(brd);
	// 	return MoveReceiver::PerfT<false, status, 18>(brd, alpha, beta);
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
