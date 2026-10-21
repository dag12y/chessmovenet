import { useEffect, useMemo, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const GLYPHS = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

export default function App() {
  const [game, setGame] = useState(null);
  const [selected, setSelected] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState(false);

  useEffect(() => {
    request("/api/health")
      .then(() => setApiReady(true))
      .catch((err) => setError(`Cannot reach the game server: ${err.message}`));
  }, []);

  const legalFromSelected = useMemo(
    () => (game?.legal_moves || []).filter((move) => move.slice(0, 2) === selected),
    [game, selected],
  );

  async function newGame() {
    setBusy(true);
    setError("");
    setSelected(null);
    try {
      setGame(await request("/api/games", { method: "POST" }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function submitMove(uci) {
    if (!game || busy) return;
    setBusy(true);
    setError("");
    setSelected(null);
    try {
      const next = await request(`/api/games/${game.game_id}/moves`, {
        method: "POST",
        body: JSON.stringify({ uci }),
      });
      setGame(next);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function onSquareClick(square, piece) {
    if (!game || busy || game.game_over) return;
    if (selected && square !== selected) {
      const candidates = legalFromSelected.filter((move) => move.slice(2, 4) === square);
      if (candidates.length) {
        const move = candidates.find((candidate) => candidate.endsWith("q")) || candidates[0];
        submitMove(move);
        return;
      }
    }
    if (piece && piece === piece.toUpperCase() && game.turn === "white") {
      setSelected(square);
    } else {
      setSelected(null);
    }
  }

  const legalTargets = new Set(legalFromSelected.map((move) => move.slice(2, 4)));

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="ChessMoveNet home">
          <span className="brand-mark">♞</span>
          <span>Chess<span className="brand-light">MoveNet</span></span>
        </a>
        <div className={`server-indicator ${apiReady ? "online" : ""}`}>
          <span /> {apiReady ? "Model server connected" : "Connecting to server"}
        </div>
      </header>

      <section className="intro" id="top">
        <div>
          <p className="eyebrow">NUMPY NEURAL NETWORK · LIVE</p>
          <h1>Take your seat.</h1>
          <p className="subhead">You play White. ChessMoveNet answers with its highest-scoring legal move.</p>
        </div>
        <button className="new-game top-new" onClick={newGame} disabled={!apiReady || busy}>
          <span>＋</span> New game
        </button>
      </section>

      <section className="play-layout">
        <div className="board-column">
          <div className="board-frame">
            <div className="board" role="grid" aria-label="Chess board">
              {game?.board.flatMap((row, rowIndex) => row.map((cell, fileIndex) => {
                const square = cell?.square || `${String.fromCharCode(97 + fileIndex)}${8 - rowIndex}`;
                const light = (rowIndex + fileIndex) % 2 === 0;
                const isSelected = selected === square;
                const isTarget = legalTargets.has(square);
                const last = game.history.at(-1)?.uci;
                const isLastMove = last && (last.slice(0, 2) === square || last.slice(2, 4) === square);
                return (
                  <button
                    role="gridcell"
                    aria-label={`${square}${cell ? ` ${cell.symbol}` : " empty"}`}
                    key={square}
                    className={`square ${light ? "light" : "dark"} ${isSelected ? "selected" : ""} ${isLastMove ? "last-move" : ""}`}
                    onClick={() => onSquareClick(square, cell?.symbol)}
                    disabled={!game || busy || game.game_over}
                  >
                    {fileIndex === 0 && <span className="coordinate rank-coordinate">{8 - rowIndex}</span>}
                    {rowIndex === 7 && <span className="coordinate file-coordinate">{String.fromCharCode(97 + fileIndex)}</span>}
                    {cell && <span className={`piece ${cell.symbol === cell.symbol.toUpperCase() ? "white-piece" : "black-piece"}`}>{GLYPHS[cell.symbol]}</span>}
                    {isTarget && <span className={cell ? "target-ring" : "target-dot"} />}
                  </button>
                );
              }))}
              {!game && <div className="board-placeholder"><span>♞</span><p>Start a game to bring the board to life.</p></div>}
            </div>
          </div>
          <div className="board-caption"><span>♙ &nbsp; You · White</span><span>♟ &nbsp; ChessMoveNet · Black</span></div>
        </div>

        <aside className="game-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">YOUR MATCH</p>
              <h2>Game room</h2>
            </div>
            <span className={`turn-badge ${game?.turn === "black" ? "thinking" : ""}`}>
              {busy ? "Thinking…" : game ? game.status : "Ready"}
            </span>
          </div>

          <button className="new-game panel-new" onClick={newGame} disabled={!apiReady || busy}>
            <span>＋</span> {game ? "Start new game" : "Start a game"}
          </button>

          {error && <div className="error-box">{error}</div>}
          {game?.in_check && !game.game_over && <div className="check-banner">Your king is in check.</div>}
          {game?.game_over && <div className="result-banner">{game.status}</div>}

          <div className="move-section">
            <div className="move-title"><span>MOVE HISTORY</span><span>{game?.history.length || 0} plies</span></div>
            <div className="move-list" aria-live="polite">
              {game?.history.length ? game.history.map((entry, index) => (
                <div className={`move-row ${entry.player === "You" ? "human-move" : "model-move"}`} key={`${entry.uci}-${index}`}>
                  <span className="move-number">{Math.floor(index / 2) + 1}{index % 2 === 0 ? "." : "…"}</span>
                  <span>{entry.san}</span>
                  <span className="move-player">{entry.player === "You" ? "YOU" : "NET"}</span>
                </div>
              )) : <div className="empty-history">Your moves will appear here.</div>}
            </div>
          </div>

          <div className="how-it-plays">
            <span className="info-icon">i</span>
            <p>The model scores every move, then plays its highest-scoring legal option.</p>
          </div>
        </aside>
      </section>
      <footer>Built with Python, NumPy, FastAPI, React &amp; Vite <span>·</span> The model plays from learned examples, not engine search.</footer>
    </main>
  );
}
