"""Single-device training loop.

Reads memmap'd uint16 shards, trains with AdamW + warmup/cosine schedule,
bf16 autocast on CUDA, and appends metrics to <out-dir>/log.csv for plotting.

    uv run python -m woodpusher.train --preset smoke --data-dir data/smoke
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .configs import PRESETS
from .model import ModelConfig, Transformer


def get_batch(data, batch_size, block_size, device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(step, max_steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * peak + 0.45 * peak * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def val_loss(model, data, batch_size, block_size, device, ctx, iters):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--out-dir", default=None, help="default: runs/<preset>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="override preset token budget")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--val-interval", type=int, default=250)
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument("--ckpt-interval", type=int, default=1000)
    ap.add_argument("--snapshot-interval", type=int, default=0,
                    help="also keep a weights-only snapshot every N steps (0 = off); "
                         "never overwritten, for probing training dynamics")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    batch_size = args.batch_size or preset.batch_size
    grad_accum = args.grad_accum or preset.grad_accum
    peak_lr = args.lr or preset.lr
    out_dir = Path(args.out_dir or f"runs/{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    cfg = ModelConfig(
        vocab_size=meta["vocab_size"],
        block_size=preset.block_size,
        n_layer=preset.n_layer,
        n_head=preset.n_head,
        n_embd=preset.n_embd,
    )
    model = Transformer(cfg).to(args.device)

    tokens_per_step = batch_size * preset.block_size * grad_accum
    max_steps = args.max_steps or max(1, preset.target_tokens // tokens_per_step)
    warmup = max(100, max_steps // 50)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95), fused=args.device == "cuda",
    )
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device == "cuda"
        else nullcontext()
    )

    start_step, best_val = 0, float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at step {start_step}")

    (out_dir / "run_meta.json").write_text(json.dumps({
        "preset": args.preset,
        "params": model.num_params(),
        "model_config": asdict(cfg),
        "max_steps": max_steps,
        "tokens_per_step": tokens_per_step,
    }, indent=2))

    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,tokens,train_loss,val_loss,lr,tok_per_s\n")

    def save(path, step):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(cfg),
            "step": step,
            "best_val": best_val,
        }, path)

    def save_snapshot(step):
        # weights-only (no optimizer): for post-hoc probing, not resuming
        torch.save({
            "model": model.state_dict(),
            "model_config": asdict(cfg),
            "step": step,
        }, out_dir / f"snap_{step:06d}.pt")

    print(f"preset={args.preset} params={model.num_params():,} device={args.device}")
    print(f"steps={max_steps:,} tokens/step={tokens_per_step:,} "
          f"total tokens={max_steps * tokens_per_step:,} (data has {len(train_data):,})")

    model.train()
    running_loss, t_last, tokens_since = None, time.time(), 0
    for step in range(start_step, max_steps):
        lr = lr_at(step, max_steps, peak_lr, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, preset.block_size, args.device)
            with ctx:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_item = loss.item()
        running_loss = loss_item if running_loss is None else 0.9 * running_loss + 0.1 * loss_item
        tokens_since += tokens_per_step

        is_val_step = (step + 1) % args.val_interval == 0 or step + 1 == max_steps
        if (step + 1) % args.log_interval == 0 or is_val_step:
            dt = time.time() - t_last
            tok_per_s = tokens_since / dt if dt > 0 else 0.0
            t_last, tokens_since = time.time(), 0
            vl = ""
            if is_val_step:
                vl = val_loss(model, val_data, batch_size, preset.block_size,
                              args.device, ctx, args.val_iters)
                if vl < best_val:
                    best_val = vl
                    save(out_dir / "best.pt", step + 1)
                vl = f"{vl:.4f}"
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, (step + 1) * tokens_per_step,
                     f"{running_loss:.4f}", vl, f"{lr:.2e}", f"{tok_per_s:.0f}"])
            print(f"step {step + 1:>6}/{max_steps} loss {running_loss:.4f} "
                  f"{'val ' + vl + ' ' if vl else ''}lr {lr:.2e} {tok_per_s:,.0f} tok/s")

        if (step + 1) % args.ckpt_interval == 0 or step + 1 == max_steps:
            save(ckpt_path, step + 1)
        if args.snapshot_interval and (step + 1) % args.snapshot_interval == 0:
            save_snapshot(step + 1)

    print(f"done. best val loss {best_val:.4f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
