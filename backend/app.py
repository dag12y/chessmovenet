"""FastAPI service that lets a person play White against ChessMoveNet."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chessmovenet.checkpoint import load_checkpoint
from chessmovenet.encode import encode_board
from chessmovenet.model import ChessMoveNet
from chessmovenet.moves import move_to_label


DEFAULT_CHECKPOINT = Path("artifacts/checkpoint_latest.pt")
CHECKPOINT_PATH = Path(os.environ.get("CHESSMOVENET_CHECKPOINT", DEFAULT_CHECKPOINT))


def load_model(path: Path) -> ChessMoveNet:
    """Load a checkpoint when present, otherwise expose a clear API error."""
    if not path.is_file():
        raise RuntimeError(
            f"Model checkpoint not found: {path}. Train a model first or set "
            "CHESSMOVENET_CHECKPOINT to a checkpoint path."
        )
    model = load_checkpoint(path, map_location="cpu")
    if model.input_size != 781 or model.output_size != 4272:
        raise RuntimeError(
            f"Checkpoint has incompatible dimensions: {model.input_size} inputs, "
            f"{model.output_size} outputs; expected 781 and 4272."
        )
    return model


MODEL = load_model(CHECKPOINT_PATH)


@dataclass
class Game:
    board: chess.Board = field(default_factory=chess.Board)
    history: list[dict[str, str]] = field(default_factory=list)


GAMES: dict[str, Game] = {}

app = FastAPI(title="ChessMoveNet", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class MoveRequest(BaseModel):
    uci: str


def _game(game_id: str) -> Game:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


def _board_rows(board: chess.Board) -> list[list[dict[str, str] | None]]:
    """Return rank 8 to rank 1, file a to h, for a simple board UI."""
    rows: list[list[dict[str, str] | None]] = []
    for rank in range(7, -1, -1):
        row = []
        for file_index in range(8):
            square = chess.square(file_index, rank)
            piece = board.piece_at(square)
            row.append(
                None
                if piece is None
                else {"symbol": piece.symbol(), "square": chess.square_name(square)}
            )
        rows.append(row)
    return rows


def _game_status(board: chess.Board) -> tuple[bool, str]:
    if not board.is_game_over(claim_draw=True):
        return False, "In progress"
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return True, f"Checkmate — {winner} wins"
    if board.is_stalemate():
        return True, "Draw by stalemate"
    if board.is_insufficient_material():
        return True, "Draw by insufficient material"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return True, "Draw by the fifty-move rule"
    if board.is_repetition(3):
        return True, "Draw by repetition"
    return True, "Draw"


def _state(game_id: str, game: Game) -> dict:
    over, status = _game_status(game.board)
    human_turn = game.board.turn == chess.WHITE and not over
    return {
        "game_id": game_id,
        "fen": game.board.fen(),
        "board": _board_rows(game.board),
        "turn": "white" if game.board.turn == chess.WHITE else "black",
        "legal_moves": [move.uci() for move in game.board.legal_moves]
        if human_turn
        else [],
        "history": game.history,
        "game_over": over,
        "status": status,
        "in_check": game.board.is_check(),
    }


def _model_move(board: chess.Board) -> chess.Move:
    """Choose the highest raw model score among this position's legal moves."""
    logits = MODEL.forward(encode_board(board))
    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError("The model was asked to move in a terminal position")
    labels = np.fromiter((move_to_label(move) for move in legal), dtype=np.int64)
    return legal[int(np.argmax(logits[labels]))]


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "checkpoint": str(CHECKPOINT_PATH),
        "input_size": MODEL.input_size,
        "output_size": MODEL.output_size,
        "activation": MODEL.activation,
    }


@app.post("/api/games")
def create_game() -> dict:
    game_id = uuid.uuid4().hex
    game = Game()
    GAMES[game_id] = game
    return _state(game_id, game)


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    return _state(game_id, _game(game_id))


@app.post("/api/games/{game_id}/moves")
def play_move(game_id: str, request: MoveRequest) -> dict:
    game = _game(game_id)
    board = game.board
    over, _ = _game_status(board)
    if over:
        raise HTTPException(status_code=409, detail="This game is already over")
    if board.turn != chess.WHITE:
        raise HTTPException(status_code=409, detail="It is the model's turn")
    try:
        move = chess.Move.from_uci(request.uci)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Move must be valid UCI notation") from exc
    if move not in board.legal_moves:
        raise HTTPException(status_code=422, detail="That move is not legal")

    human_san = board.san(move)
    board.push(move)
    game.history.append({"player": "You", "uci": move.uci(), "san": human_san})

    if not board.is_game_over(claim_draw=True):
        reply = _model_move(board)
        model_san = board.san(reply)
        board.push(reply)
        game.history.append(
            {"player": "ChessMoveNet", "uci": reply.uci(), "san": model_san}
        )

    return _state(game_id, game)
