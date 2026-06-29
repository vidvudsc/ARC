#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import socket
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from arc.checkpoint import load_checkpoint
from arc.config import ArcConfig, load_config
from arc.model import ArcModel
from arc.tokenizer import ArcTokenizer


MODEL: ArcModel | None = None
TOKENIZER: ArcTokenizer | None = None
DEVICE: torch.device | None = None
CFG: ArcConfig | None = None
ARGS: argparse.Namespace | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    data = torch.from_numpy(array).permute(2, 0, 1)
    return (data - 0.5) / 0.5


def build_inputs(
    tokenizer: ArcTokenizer,
    cfg: ArcConfig,
    prompt: str,
    image: Image.Image | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    prompt_ids = tokenizer.encode(prompt, add_bos=image is None, add_eos=False)
    if image is None:
        ids = prompt_ids
        image_tensor = None
        image_positions = None
    else:
        prefix = [tokenizer.image_start_id] + [tokenizer.pad_id] * cfg.vision_resampler_tokens + [
            tokenizer.image_end_id
        ]
        ids = prefix + prompt_ids
        image_tensor = normalize_image(image, cfg.image_size).unsqueeze(0).to(device)
        image_positions = torch.tensor([1], dtype=torch.long, device=device)

    if len(ids) >= cfg.max_seq_len:
        ids = ids[-(cfg.max_seq_len - 1) :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return input_ids, image_tensor, image_positions


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    vals, idx = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, idx, vals)
    return filtered


def dtype_for_autocast(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def generate_events(prompt: str, image: Image.Image | None) -> Iterable[dict[str, object]]:
    assert MODEL is not None and TOKENIZER is not None and DEVICE is not None and CFG is not None and ARGS is not None
    input_ids, image_tensor, image_positions = build_inputs(TOKENIZER, CFG, prompt, image, DEVICE)
    eos_id = TOKENIZER.eos_id
    stop_ids = {TOKENIZER.pad_id, TOKENIZER.image_start_id, TOKENIZER.image_end_id}

    emitted_ids: list[int] = []
    for _ in range(ARGS.max_new_tokens):
        if input_ids.shape[1] >= CFG.max_seq_len:
            break

        autocast_enabled = DEVICE.type == "cuda" and ARGS.dtype != "float32"
        with torch.autocast(
            device_type="cuda",
            dtype=dtype_for_autocast(ARGS.dtype),
            enabled=autocast_enabled,
        ):
            logits, _ = MODEL(input_ids, images=image_tensor, image_insert_positions=image_positions)

        next_logits = logits[:, -1, :] / max(float(ARGS.temperature), 1e-6)
        next_logits = apply_top_k(next_logits, int(ARGS.top_k))
        probs = F.softmax(next_logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, min(5, probs.shape[-1]), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if token_id == eos_id or token_id in stop_ids:
            break

        emitted_ids.append(token_id)
        text = TOKENIZER.decode([token_id], skip_special_tokens=True)
        if not text:
            continue

        yield {
            "type": "token",
            "text": text,
            "token_id": token_id,
            "prob": float(probs[0, token_id].detach().cpu()),
            "top": [
                {
                    "token": TOKENIZER.decode([int(idx)], skip_special_tokens=True),
                    "prob": float(prob),
                }
                for prob, idx in zip(top_vals[0].detach().cpu().tolist(), top_idx[0].detach().cpu().tolist())
            ],
        }

    yield {
        "type": "done",
        "text": TOKENIZER.decode(emitted_ids, skip_special_tokens=True),
        "tokens": len(emitted_ids),
    }


def load_model(args: argparse.Namespace) -> tuple[ArcModel, ArcTokenizer, ArcConfig]:
    tokenizer = ArcTokenizer.from_dir(args.tokenizer_dir)
    if args.checkpoint:
        # Keep checkpoint deserialization on CPU. Inference only needs the model
        # weights, and training checkpoints also contain optimizer state that is
        # wasteful to materialize directly on MPS/CUDA.
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = ArcConfig(**ckpt.get("config", asdict(load_config(args.config))))
    elif args.allow_random_init:
        ckpt = None
        cfg = load_config(args.config)
    else:
        raise SystemExit("--checkpoint is required unless --allow_random_init is set")

    model = ArcModel(cfg, pad_id=tokenizer.pad_id)
    if ckpt is not None:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(args.device)
    model.eval()
    return model, tokenizer, cfg


def chunk_write(handler: BaseHTTPRequestHandler, data: str) -> None:
    raw = data.encode("utf-8")
    handler.wfile.write(f"{len(raw):x}\r\n".encode("ascii"))
    handler.wfile.write(raw)
    handler.wfile.write(b"\r\n")
    handler.wfile.flush()


def finish_chunks(handler: BaseHTTPRequestHandler) -> None:
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    server_version = "ArcMini/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        assert ARGS is not None
        if self.path in {"/", "/index.html"}:
            path = Path(ARGS.index)
            if not path.exists():
                self.send_error(404, f"Missing {path}")
                return
            body = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        import cgi

        assert ARGS is not None and DEVICE is not None and CFG is not None
        if self.path != "/api/generate":
            self.send_error(404)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        prompt = form.getfirst("prompt", "A photo of ")
        image = None
        if "image" in form and getattr(form["image"], "filename", ""):
            image = Image.open(form["image"].file).convert("RGB")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            chunk_write(
                self,
                json.dumps(
                    {
                        "type": "meta",
                        "model": CFG.model_name,
                        "checkpoint": str(ARGS.checkpoint) if ARGS.checkpoint else "random-init",
                        "device": str(DEVICE),
                        "image": image is not None,
                        "max_new_tokens": ARGS.max_new_tokens,
                        "temperature": ARGS.temperature,
                        "top_k": ARGS.top_k,
                    }
                )
                + "\n",
            )
            for event in generate_events(prompt, image):
                chunk_write(self, json.dumps(event, ensure_ascii=False) + "\n")
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            print("client disconnected during generation")
        except Exception as exc:
            chunk_write(self, json.dumps({"type": "error", "message": str(exc)}) + "\n")
        finally:
            try:
                finish_chunks(self)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_artifacts = repo_root / "downloads" / "mini_mixed_1b" / "arc_mini_artifacts"
    default_checkpoint = default_artifacts / "arc_mini_mixed_last.pt"
    default_tokenizer = default_artifacts / "tokenizer_16k"

    parser = argparse.ArgumentParser(description="Arc-Mini text/image inference")
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint) if default_checkpoint.exists() else None,
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--tokenizer_dir",
        default=str(default_tokenizer) if default_tokenizer.exists() else None,
    )
    parser.add_argument("--prompt", default="A photo of ")
    parser.add_argument("--image", default=None)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--allow_random_init", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--index", default="index.html")
    args = parser.parse_args()
    if args.tokenizer_dir is None:
        raise SystemExit("--tokenizer_dir is required unless the default downloaded tokenizer exists")
    return args


def main() -> None:
    global MODEL, TOKENIZER, DEVICE, CFG, ARGS
    ARGS = parse_args()
    DEVICE = torch.device(ARGS.device)
    MODEL, TOKENIZER, CFG = load_model(ARGS)

    if ARGS.serve:
        print(f"Loaded {CFG.model_name} on {DEVICE}")
        print(f"Serving http://{ARGS.host}:{ARGS.port}")
        ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
        return

    image = Image.open(ARGS.image).convert("RGB") if ARGS.image else None
    final_text = ""
    for event in generate_events(ARGS.prompt, image):
        if event["type"] == "token":
            piece = str(event["text"])
            final_text += piece
            print(piece, end="", flush=True)
    print()
    if not final_text:
        print("[no text generated]")


if __name__ == "__main__":
    main()
