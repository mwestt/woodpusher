"""Local web UI: watch training live and play any checkpoint, on one page.

    uv run python -m woodpusher.web            # then open http://localhost:8000

Zero extra dependencies (stdlib server, hand-rolled board, canvas chart).
Checkpoints are hot-reloaded when their file changes, so a model can be
watched — and played — while it trains.
"""

import argparse
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import torch

from .evals.common import load_model, pick_move
from .tokenizer import Tokenizer

TOK = Tokenizer()
INFER_LOCK = threading.Lock()
_model_cache: dict[str, tuple[float, object]] = {}


def get_model(run_dir: Path, device: str):
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        ckpt = run_dir / "best.pt"
    mtime = ckpt.stat().st_mtime
    key = str(ckpt)
    with INFER_LOCK:
        cached = _model_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    model, _ = load_model(ckpt, device)
    with INFER_LOCK:
        _model_cache[key] = (mtime, model)
    return model


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
        elif url.path == "/api/log":
            run = parse_qs(url.query).get("run", [""])[0]
            log_path = self.runs_dir / run / "log.csv"
            rows = []
            if run and log_path.exists():
                with open(log_path, newline="") as f:
                    for r in csv.DictReader(f):
                        rows.append({
                            "step": int(r["step"]),
                            "tokens": int(r["tokens"]),
                            "train_loss": float(r["train_loss"]),
                            "val_loss": float(r["val_loss"]) if r["val_loss"] else None,
                            "tok_per_s": float(r["tok_per_s"]),
                        })
            self._json({"rows": rows})
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

        error = note = None
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
                board.push(mv)
                moves.append(mv.uci())

        if error is None and not board.is_game_over():
            model_turn = selfplay or ((board.turn == chess.WHITE) != user_white)
            if model_turn:
                if selfplay:
                    welo = belo = model_elo
                else:
                    welo = your_elo if user_white else model_elo
                    belo = model_elo if user_white else your_elo
                ids = TOK.prefix_ids(welo, belo) + [TOK.move_id(u) for u in moves]
                model = get_model(self.runs_dir / req["run"], self.device)
                with INFER_LOCK:
                    mv, raw_id = pick_move(model, TOK, ids, board, self.device, temperature)
                raw_ok = False
                if TOK.is_move_id(raw_id):
                    try:
                        raw_ok = chess.Move.from_uci(TOK.tokens[raw_id]) in board.legal_moves
                    except ValueError:
                        pass
                if not raw_ok:
                    note = f"raw top-1 was illegal: {TOK.tokens[raw_id]}"
                board.push(mv)
                moves.append(mv.uci())

        replay, sans = chess.Board(), []
        for u in moves:
            m = chess.Move.from_uci(u)
            sans.append(replay.san(m))
            replay.push(m)

        return {
            "error": error,
            "note": note,
            "fen": board.fen(),
            "moves": moves,
            "sans": sans,
            "legal": [m.uci() for m in board.legal_moves],
            "turn": "w" if board.turn == chess.WHITE else "b",
            "game_over": board.is_game_over(),
            "result": board.result(claim_draw=True) if board.is_game_over() else None,
        }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>woodpusher</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e1e22; color: #ddd;
         margin: 0; display: flex; gap: 24px; padding: 20px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0 0 10px; } h2 { font-size: 14px; margin: 14px 0 6px; color: #aaa; }
  #board { display: grid; grid-template-columns: repeat(8, 52px); user-select: none;
           border: 2px solid #444; width: fit-content; }
  .sq { width: 52px; height: 52px; font-size: 38px; line-height: 52px; text-align: center; cursor: pointer; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .sel { outline: 3px solid #e8a33d; outline-offset: -3px; }
  .tgt { box-shadow: inset 0 0 0 4px rgba(60,120,60,.55); }
  .wp { color: #fafafa; text-shadow: 0 0 2px #000; } .bp { color: #1a1a1a; }
  select, input, button { background: #2b2b31; color: #ddd; border: 1px solid #555;
                          border-radius: 4px; padding: 4px 8px; margin: 2px; }
  button { cursor: pointer; } button:hover { background: #3a3a42; }
  #status { margin-top: 8px; min-height: 20px; color: #e8a33d; }
  #notes { font-size: 12px; color: #888; max-height: 90px; overflow-y: auto; }
  #sans { font-size: 13px; max-width: 420px; max-height: 120px; overflow-y: auto; color: #bbb; }
  canvas { background: #26262c; border: 1px solid #444; }
  #trainstats { font-size: 13px; color: #9c9; }
  .panel { min-width: 430px; }
</style></head><body>

<div class="panel">
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
  <h2>model notes (raw top-1 legality)</h2>
  <div id="notes"></div>
</div>

<div class="panel">
  <h2>training loss (polls log.csv of the selected run)</h2>
  <canvas id="chart" width="460" height="300"></canvas>
  <div id="trainstats"></div>
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
  if (d.note) addNote(S.moves.length + ". " + d.note);
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
  document.getElementById("notes").innerHTML = "";
  document.getElementById("sans").textContent = "";
  renderBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  if (selfplay) {
    selfplayTimer = setInterval(() => { if (!S.busy && !S.over) sendMove(null, true); }, 500);
  } else if (!S.userWhite) {
    sendMove(null, false);   // model is white: it opens
  } else {
    setStatus("your move");
  }
}

function stopSelfplay() { if (selfplayTimer) clearInterval(selfplayTimer); selfplayTimer = null; S.selfplay = false; }
function setStatus(t) { document.getElementById("status").textContent = t; }
function addNote(t) { const n = document.getElementById("notes"); n.innerHTML = t + "<br>" + n.innerHTML; }

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

async function pollLog() {
  if (!S.run) return;
  const d = await (await fetch("/api/log?run=" + S.run)).json();
  const rows = d.rows;
  const c = document.getElementById("chart"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (!rows.length) return;
  const losses = rows.map(r => r.train_loss).concat(rows.filter(r => r.val_loss).map(r => r.val_loss));
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const x = s => 40 + (s / rows[rows.length - 1].step) * (c.width - 55);
  const y = l => 10 + (1 - (l - lo) / (hi - lo + 1e-9)) * (c.height - 40);
  g.strokeStyle = "#666"; g.beginPath();
  rows.forEach((r, i) => i ? g.lineTo(x(r.step), y(r.train_loss)) : g.moveTo(x(r.step), y(r.train_loss)));
  g.stroke();
  g.fillStyle = "#e8a33d";
  for (const r of rows) if (r.val_loss) { g.beginPath(); g.arc(x(r.step), y(r.val_loss), 3, 0, 7); g.fill(); }
  g.fillStyle = "#999"; g.font = "11px sans-serif";
  g.fillText(hi.toFixed(2), 4, 16); g.fillText(lo.toFixed(2), 4, c.height - 34);
  g.fillText("step " + rows[rows.length - 1].step, c.width - 80, c.height - 8);
  const last = rows[rows.length - 1];
  document.getElementById("trainstats").textContent =
    "step " + last.step + "  train " + last.train_loss.toFixed(4) +
    (last.val_loss ? "  val " + last.val_loss.toFixed(4) : "") +
    "  " + Math.round(last.tok_per_s).toLocaleString() + " tok/s";
}

loadRuns().then(() => { renderBoard(null); pollLog(); });
setInterval(pollLog, 3000);
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
