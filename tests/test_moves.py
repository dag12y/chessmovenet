"""Tests for chessmovenet.moves."""

import numpy as np

import chess

from chessmovenet.moves import (
    OUTPUT_SIZE,
    encode_moves,
    index_to_move,
    legal_mask,
    move_to_index,
)


def test_output_size_constant():
    assert OUTPUT_SIZE == 4096


def test_move_to_index_d7d5():
    assert move_to_index("d7d5") == 3299


def test_e2e4_round_trip():
    assert index_to_move(move_to_index("e2e4")) == "e2e4"


def test_promotion_suffix_ignored():
    assert move_to_index("e7e8q") == move_to_index("e7e8")


def test_start_legal_mask_has_20_ones():
    mask = legal_mask(chess.Board().fen())
    assert mask.shape == (OUTPUT_SIZE,)
    assert mask.dtype == np.float32
    assert int(mask.sum()) == 20


def test_played_move_is_in_legal_mask():
    """The move actually played must always be a legal class in its position."""
    data = np.load("data/processed/train.npz", allow_pickle=True)
    fens = data["fen"][:1000]
    moves = data["uci"][:1000]
    for fen, uci in zip(fens, moves):
        mask = legal_mask(fen)
        idx = move_to_index(uci)
        assert mask[idx] == 1.0, f"illegal move {uci} in {fen}"


def test_encode_moves_shape_and_type():
    ucis = ["e2e4", "d7d5", "e7e8q"]
    encoded = encode_moves(ucis)
    assert encoded.shape == (3,)
    assert encoded.dtype == np.int64
    assert encoded[0] == move_to_index("e2e4")
    assert encoded[1] == move_to_index("d7d5")
    assert encoded[2] == move_to_index("e7e8")
