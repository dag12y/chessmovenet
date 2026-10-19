"""Tests for chessmovenet.encode."""

import chess
import numpy as np

from chessmovenet.encode import INPUT_SIZE, encode_fens, fen_to_vector

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_vector_shape_and_dtype():
    vector = fen_to_vector(START_FEN)
    assert vector.shape == (INPUT_SIZE,)
    assert vector.dtype == np.float32


def test_starting_position_has_32_ones():
    vector = fen_to_vector(START_FEN)
    assert int(vector[:768].sum()) == 32


def test_white_pawn_on_e2():
    vector = fen_to_vector(START_FEN)
    # White pawn: plane 0, square e2.
    assert vector[0 * 64 + chess.E2] == 1.0


def test_black_king_on_e8():
    vector = fen_to_vector(START_FEN)
    # Black king: plane 6 + (6 - 1) = 11, square e8.
    assert vector[11 * 64 + chess.E8] == 1.0


def test_turn_flag():
    # White to move at the start.
    assert fen_to_vector(START_FEN)[768] == 1.0
    # After 1. e4 it is Black's turn.
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    assert fen_to_vector(board.fen())[768] == 0.0


def test_castling_flags_start_and_lost():
    vector = fen_to_vector(START_FEN)
    assert list(vector[769:773]) == [1.0, 1.0, 1.0, 1.0]
    # White has lost castling rights but Black keeps both: castling = "kq".
    no_castle_white = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w kq - 0 1"
    vector2 = fen_to_vector(no_castle_white)
    assert vector2[769] == 0.0  # White kingside lost
    assert vector2[770] == 0.0  # White queenside lost
    assert vector2[771] == 1.0  # Black kingside still available
    assert vector2[772] == 1.0  # Black queenside still available


def test_encode_fens_shape():
    fens = [
        START_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    encoded = encode_fens(fens)
    assert encoded.shape == (3, INPUT_SIZE)
    assert encoded.dtype == np.float32
