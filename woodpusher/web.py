"""Local web UI: play any checkpoint and watch its move distribution live.

    uv run python -m woodpusher.web            # then open http://localhost:8000

Zero extra dependencies (stdlib server, hand-rolled board). Centered board
with a scrolling prediction feed: for every position — before your moves and
the model's — the raw top-k next-move distribution, with the played move
highlighted and illegal candidates flagged. Checkpoints are hot-reloaded
when their file changes.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import torch

from .evals.common import next_logits, pick_move, topk_report
from .model import ModelConfig, Transformer
from .tokenizer import Tokenizer

TOK = Tokenizer()
INFER_LOCK = threading.Lock()
_model_cache: dict[str, tuple[float, object, int]] = {}


def get_model(run_dir: Path, device: str):
    ckpt_path = run_dir / "ckpt.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best.pt"
    mtime = ckpt_path.stat().st_mtime
    key = str(ckpt_path)
    with INFER_LOCK:
        cached = _model_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1], cached[2], mtime
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = int(ckpt.get("step", 0))
    with INFER_LOCK:
        _model_cache[key] = (mtime, model, step)
    return model, step, mtime


def pred_entry(logits, board, played_uci, mover, ply):
    """Feed entry for one position: top-k of the raw distribution plus where
    the actually-played move ranked. Computed before the move is pushed."""
    probs, legal_mass, top = topk_report(logits, TOK, board)
    played_id = TOK.move_id(played_uci)
    return {
        "ply": ply,
        "mover": mover,
        "san": board.san(chess.Move.from_uci(played_uci)),
        "top": [
            {"move": t["move"], "prob": t["prob"], "legal": t["legal"], "played": t["id"] == played_id}
            for t in top
        ],
        "legal_mass": legal_mass,
        "played_prob": float(probs[played_id]),
        "played_rank": int((probs > probs[played_id]).sum()) + 1,
    }


class Handler(BaseHTTPRequestHandler):
    runs_dir = Path("runs")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/runs":
            runs = []
            for d in sorted(self.runs_dir.glob("*")):
                if not ((d / "ckpt.pt").exists() or (d / "best.pt").exists()):
                    continue
                params = None
                meta_path = d / "run_meta.json"
                if meta_path.exists():
                    params = json.loads(meta_path.read_text()).get("params")
                runs.append({"name": d.name, "params": params})
            self._json(runs)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/move":
            self._json({"error": "not found"}, 404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._json(self._move(req))
        except Exception as e:  # surface errors to the page instead of a hang
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _move(self, req):
        board = chess.Board()
        moves = list(req.get("moves", []))
        for u in moves:
            board.push(chess.Move.from_uci(u))
        user_white = bool(req.get("user_white", True))
        selfplay = bool(req.get("selfplay", False))
        model_elo = int(req.get("model_elo", 1800))
        your_elo = int(req.get("your_elo", 1600))
        temperature = float(req.get("temperature", 0.5))

        if selfplay:
            welo = belo = model_elo
        else:
            welo = your_elo if user_white else model_elo
            belo = model_elo if user_white else your_elo
        prefix = TOK.prefix_ids(welo, belo)
        model = None

        def logits_now():
            nonlocal model
            if model is None:
                model, _, _ = get_model(self.runs_dir / req["run"], self.device)
            ids = prefix + [TOK.move_id(u) for u in moves]
            with INFER_LOCK:
                return next_logits(model, ids, self.device)

        error = None
        preds = []
        user_move = req.get("user_move")
        if user_move and not board.is_game_over():
            mv = None
            for candidate in (user_move, user_move + "q"):  # bare promotion -> queen
                try:
                    m = chess.Move.from_uci(candidate)
                except ValueError:
                    continue
                if m in board.legal_moves:
                    mv = m
                    break
            if mv is None:
                error = f"illegal move: {user_move}"
            else:
                preds.append(pred_entry(logits_now(), board, mv.uci(), "you", len(moves)))
                board.push(mv)
                moves.append(mv.uci())

        if error is None and not board.is_game_over():
            model_turn = selfplay or ((board.turn == chess.WHITE) != user_white)
            if model_turn:
                logits = logits_now()
                with INFER_LOCK:
                    mv, _ = pick_move(model, TOK, [], board, self.device, temperature, logits=logits)
                preds.append(pred_entry(logits, board, mv.uci(), "model", len(moves)))
                board.push(mv)
                moves.append(mv.uci())

        replay, sans = chess.Board(), []
        for u in moves:
            m = chess.Move.from_uci(u)
            sans.append(replay.san(m))
            replay.push(m)

        return {
            "error": error,
            "fen": board.fen(),
            "moves": moves,
            "sans": sans,
            "preds": preds,
            "legal": [m.uci() for m in board.legal_moves],
            "turn": "w" if board.turn == chess.WHITE else "b",
            "game_over": board.is_game_over(),
            "result": board.result(claim_draw=True) if board.is_game_over() else None,
        }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>woodpusher</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e1e22; color: #ddd; margin: 0;
         display: flex; justify-content: center; align-items: flex-start;
         gap: 28px; padding: 20px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0 0 10px; } h2 { font-size: 14px; margin: 0 0 8px; color: #aaa; }
  #board { display: grid; grid-template-columns: repeat(8, 52px); user-select: none;
           border: 2px solid #444; width: fit-content; margin-top: 8px; }
  .sq { width: 52px; height: 52px; font-size: 38px; line-height: 52px; text-align: center; cursor: pointer; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .sel { outline: 3px solid #e8a33d; outline-offset: -3px; }
  .tgt { box-shadow: inset 0 0 0 4px rgba(60,120,60,.55); }
  .wp { color: #fafafa; text-shadow: 0 0 2px #000; } .bp { color: #1a1a1a; }
  select, input, button { background: #2b2b31; color: #ddd; border: 1px solid #555;
                          border-radius: 4px; padding: 4px 8px; margin: 2px; }
  button { cursor: pointer; } button:hover { background: #3a3a42; }
  #status { margin-top: 8px; min-height: 20px; color: #e8a33d; }
  #sans { font-size: 13px; max-width: 432px; max-height: 100px; overflow-y: auto; color: #bbb; }
  #feedpanel { width: 340px; }
  #feed { max-height: calc(100vh - 90px); overflow-y: auto; }
  .entry { border-bottom: 1px solid #333; padding: 8px 4px; }
  .ehead { font-size: 13px; margin-bottom: 4px; }
  .who { color: #888; font-size: 11px; margin: 0 6px; text-transform: uppercase; }
  .row { display: flex; align-items: center; gap: 6px; font-size: 12px; margin: 1px 0;
         padding: 0 2px; border-radius: 3px; }
  .row.played { background: rgba(232,163,61,.16); }
  .mv { width: 62px; } .illegal { color: #e66; }
  .bar { height: 10px; background: #5a9; } .pct { color: #888; }
</style></head><body>

<div>
  <h1>woodpusher</h1>
  <div>
    run <select id="run"></select>
    you play <select id="color"><option>white</option><option>black</option></select>
    model elo <input id="melo" type="number" value="1800" style="width:70px">
    your elo <input id="yelo" type="number" value="1600" style="width:70px">
    temp <input id="temp" type="number" value="0.5" step="0.1" style="width:60px">
  </div>
  <div>
    <button onclick="newGame(false)">new game vs model</button>
    <button onclick="newGame(true)">self-play</button>
    <button onclick="stopSelfplay()">stop</button>
  </div>
  <div id="board"></div>
  <div id="status">pick a run and start a game</div>
  <div id="sans"></div>
</div>

<div id="feedpanel">
  <h2>candidate moves (raw distribution, ✗ = illegal)</h2>
  <div id="feed"></div>
</div>

<script>
const GLYPH = {p:"♟", n:"♞", b:"♝", r:"♜", q:"♛", k:"♚"};
let S = {run:null, moves:[], legal:[], sel:null, userWhite:true, selfplay:false, over:true, busy:false};
let selfplayTimer = null;

function sqName(file, rank) { return "abcdefgh"[file] + (rank + 1); }

function renderBoard(fen) {
  const board = document.getElementById("board");
  board.innerHTML = "";
  const placement = (fen || "8/8/8/8/8/8/8/8 w - - 0 1").split(" ")[0].split("/");
  const grid = {};
  placement.forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) file += +ch;
      else { grid[sqName(file, 7 - i)] = ch; file++; }
    }
  });
  const ranks = S.userWhite ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];
  const files = S.userWhite ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  for (const r of ranks) for (const f of files) {
    const name = sqName(f, r);
    const div = document.createElement("div");
    div.className = "sq " + ((r + f) % 2 ? "light" : "dark");
    const piece = grid[name];
    if (piece) {
      div.textContent = GLYPH[piece.toLowerCase()];
      div.classList.add(piece === piece.toUpperCase() ? "wp" : "bp");
    }
    if (S.sel === name) div.classList.add("sel");
    if (S.sel && S.legal.some(u => u.startsWith(S.sel) && u.slice(2, 4) === name)) div.classList.add("tgt");
    div.onclick = () => clickSquare(name, piece);
    board.appendChild(div);
  }
}

function clickSquare(name, piece) {
  if (S.over || S.selfplay || S.busy) return;
  if (S.sel && S.legal.some(u => u.startsWith(S.sel) && u.slice(2, 4) === name)) {
    const move = S.sel + name;
    S.sel = null;
    sendMove(move, false);
    return;
  }
  const mine = piece && (piece === piece.toUpperCase()) === S.userWhite;
  S.sel = mine ? name : null;
  renderBoard(S.fen);
}

function pc(x) { return (x * 100).toFixed(1) + "%"; }

function renderPred(p) {
  const moveNo = Math.floor(p.ply / 2) + 1;
  const num = p.ply % 2 === 0 ? moveNo + "." : moveNo + "…";
  let html = "<div class='ehead'><b>" + num + " " + p.san + "</b>" +
    "<span class='who'>" + p.mover + "</span>" +
    "<span class='pct'>p " + pc(p.played_prob) + " · rank #" + p.played_rank +
    " · legal mass " + pc(p.legal_mass) + "</span></div>";
  for (const t of p.top) {
    html += "<div class='row" + (t.played ? " played" : "") + "'>" +
      "<span class='mv" + (t.legal ? "" : " illegal") + "'>" + t.move + (t.legal ? "" : " ✗") + "</span>" +
      "<div class='bar' style='width:" + Math.max(1, Math.round(t.prob * 180)) + "px'></div>" +
      "<span class='pct'>" + pc(t.prob) + "</span></div>";
  }
  const div = document.createElement("div");
  div.className = "entry";
  div.innerHTML = html;
  return div;
}

async function sendMove(userMove, selfplayStep) {
  if (!S.run || S.busy) return;
  S.busy = true;
  setStatus(userMove ? "thinking..." : "model to move...");
  const res = await fetch("/api/move", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      run: S.run, moves: S.moves, user_move: userMove, selfplay: selfplayStep,
      user_white: S.userWhite, model_elo: +document.getElementById("melo").value,
      your_elo: +document.getElementById("yelo").value,
      temperature: +document.getElementById("temp").value,
    })});
  const d = await res.json();
  S.busy = false;
  if (d.error) { setStatus(d.error); return; }
  S.moves = d.moves; S.legal = d.legal; S.fen = d.fen; S.over = d.game_over;
  const feed = document.getElementById("feed");
  for (const p of d.preds) feed.prepend(renderPred(p));
  document.getElementById("sans").textContent = d.sans.map((s, i) =>
    i % 2 === 0 ? Math.floor(i / 2 + 1) + ". " + s : s).join(" ");
  renderBoard(d.fen);
  setStatus(d.game_over ? "game over: " + d.result
            : (d.turn === "w" ? "white" : "black") + " to move");
  if (d.game_over) stopSelfplay();
}

function newGame(selfplay) {
  stopSelfplay();
  S.moves = []; S.legal = []; S.sel = null; S.over = false; S.selfplay = selfplay;
  S.userWhite = selfplay || document.getElementById("color").value === "white";
  S.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  document.getElementById("sans").textContent = "";
  document.getElementById("feed").innerHTML = "";
  renderBoard(S.fen);
  // hydrate from the server: fills legal moves; model replies if it's to move
  sendMove(null, false);
  if (selfplay) {
    selfplayTimer = setInterval(() => { if (!S.busy && !S.over) sendMove(null, true); }, 500);
  }
}

function stopSelfplay() { if (selfplayTimer) clearInterval(selfplayTimer); selfplayTimer = null; S.selfplay = false; }
function setStatus(t) { document.getElementById("status").textContent = t; }

async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  const sel = document.getElementById("run");
  const prev = S.run;
  sel.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.name;
    o.textContent = r.name + (r.params ? " (" + (r.params / 1e6).toFixed(1) + "M)" : "");
    sel.appendChild(o);
  }
  if (runs.length) { S.run = prev && runs.some(r => r.name === prev) ? prev : runs[0].name; sel.value = S.run; }
  sel.onchange = () => { S.run = sel.value; };
}

loadRuns().then(() => renderBoard(null));
setInterval(loadRuns, 15000);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    Handler.runs_dir = Path(args.runs_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"woodpusher ui: http://localhost:{args.port}  (device: {Handler.device})")
    server.serve_forever()


if __name__ == "__main__":
    main()
