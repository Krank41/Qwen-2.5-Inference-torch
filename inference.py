#!/usr/bin/env python3
"""
Pure PyTorch Qwen2.5-0.5B-Instruct inference.

Runtime dependencies:
  - torch
  - Python standard library

No transformers, tokenizers, huggingface_hub, accelerate, or safetensors package.

Expected model directory files:
  config.json
  model.safetensors
  vocab.json + merges.txt
  tokenizer_config.json   optional but recommended

Example:
  python qwen25_pure_torch.py --model-dir ./qwen25_05b_instruct --prompt "Write a haiku about GPUs."
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import mmap
import os
import re
import struct
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Minimal safetensors reader
# -----------------------------

_SAFE_DTYPE_TO_TORCH = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _read_safetensors_header(path: str) -> Tuple[int, dict]:
    with open(path, "rb") as f:
        first8 = f.read(8)
        if len(first8) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file.")
        header_len = struct.unpack("<Q", first8)[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def load_safetensors_into_model(model: nn.Module, model_dir: str) -> None:
    safetensor_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_paths:
        raise FileNotFoundError(f"No .safetensors files found in: {model_dir}")

    params = dict(model.named_parameters())
    loaded = set()
    unexpected = []

    with torch.no_grad():
        for path in safetensor_paths:
            header_len, header = _read_safetensors_header(path)
            data_base = 8 + header_len

            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue

                        # Hugging Face Qwen2ForCausalLM stores the transformer under
                        # the "model." prefix. This file keeps the module flatter, so
                        # map "model.layers..." -> "layers..." etc.
                        target_name = name
                        if target_name not in params and target_name.startswith("model."):
                            target_name = target_name[len("model.") :]

                        # This implementation ties lm_head to embeddings directly, so a
                        # separate lm_head.weight, if present, is intentionally ignored.
                        if name == "lm_head.weight" and target_name not in params:
                            continue

                        if target_name not in params:
                            unexpected.append(name)
                            continue

                        dtype_name = info["dtype"]
                        shape = tuple(int(x) for x in info["shape"])
                        start, end = info["data_offsets"]
                        dtype = _SAFE_DTYPE_TO_TORCH[dtype_name]
                        numel = _prod(shape)

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="The given buffer is not writable",
                                category=UserWarning,
                            )
                            tensor = torch.frombuffer(
                                mm,
                                dtype=dtype,
                                count=numel,
                                offset=data_base + int(start),
                            ).reshape(shape)

                        param = params[target_name]
                        if tuple(param.shape) != shape:
                            raise ValueError(
                                f"Shape mismatch for {name}: checkpoint {shape}, model {tuple(param.shape)}"
                            )

                        param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                        loaded.add(target_name)

                    # Make sure all asynchronous device copies finish before the mmap closes.
                    if any(p.device.type == "cuda" for p in params.values()):
                        torch.cuda.synchronize()
                finally:
                    mm.close()

    missing = [name for name in params if name not in loaded]
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} tensors from checkpoint. First missing: {sample}")

    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected checkpoint tensors. First: {unexpected[:5]}")


# -----------------------------
# Qwen2 tokenizer, pure Python
# -----------------------------

def bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    # GPT-2 / ByteLevel reversible byte-to-unicode map.
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder


def _is_letter(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("N")


def _is_space(ch: str) -> bool:
    return bool(ch) and ch.isspace()


def _is_crlf(ch: str) -> bool:
    return ch == "\r" or ch == "\n"


def qwen_pretokenize(text: str) -> List[str]:
    r"""
    Implements the Qwen2 pre-tokenization regex in stdlib Python:
      (?i:'s|'t|'re|'ve|'m|'ll|'d)
      |[^\r\n\p{L}\p{N}]?\p{L}+
      |\p{N}
      | ?[^\s\p{L}\p{N}]+[\r\n]*
      |\s*[\r\n]+
      |\s+(?!\S)
      |\s+
    """
    out: List[str] = []
    i = 0
    n = len(text)
    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")

    while i < n:
        # 1) contractions, case-insensitive
        matched = None
        for c in contractions:
            if text[i : i + len(c)].lower() == c:
                matched = text[i : i + len(c)]
                break
        if matched is not None:
            out.append(matched)
            i += len(matched)
            continue

        ch = text[i]

        # 2) optional non-CRLF/non-letter/non-number prefix + letters
        if _is_letter(ch) or (
            not _is_crlf(ch)
            and not _is_letter(ch)
            and not _is_number(ch)
            and i + 1 < n
            and _is_letter(text[i + 1])
        ):
            j = i
            if not _is_letter(text[j]):
                j += 1
            if j < n and _is_letter(text[j]):
                j += 1
                while j < n and _is_letter(text[j]):
                    j += 1
                out.append(text[i:j])
                i = j
                continue

        # 3) single numeric character
        if _is_number(ch):
            out.append(ch)
            i += 1
            continue

        # 4) optional ASCII space + punctuation/symbol run + optional CR/LF
        if (
            ch == " "
            and i + 1 < n
            and (not _is_space(text[i + 1]))
            and (not _is_letter(text[i + 1]))
            and (not _is_number(text[i + 1]))
        ):
            j = i + 1
            while j < n and (not _is_space(text[j])) and (not _is_letter(text[j])) and (not _is_number(text[j])):
                j += 1
            while j < n and _is_crlf(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        if (not _is_space(ch)) and (not _is_letter(ch)) and (not _is_number(ch)):
            j = i + 1
            while j < n and (not _is_space(text[j])) and (not _is_letter(text[j])) and (not _is_number(text[j])):
                j += 1
            while j < n and _is_crlf(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # 5) whitespace ending in a CR/LF run
        if _is_space(ch):
            j = i
            while j < n and _is_space(text[j]):
                j += 1

            last_newline = -1
            k = i
            while k < j:
                if _is_crlf(text[k]):
                    last_newline = k
                k += 1

            if last_newline >= 0:
                end = last_newline + 1
                while end < j and _is_crlf(text[end]):
                    end += 1
                out.append(text[i:end])
                i = end
                continue

            # 6/7) ordinary whitespace, including trailing whitespace
            out.append(text[i:j])
            i = j
            continue

        # Fallback: should not be reached, but keeps the tokenizer total.
        out.append(ch)
        i += 1

    return out


def get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    if len(word) < 2:
        return set()
    return set(zip(word, word[1:]))


class Qwen2TokenizerPure:
    def __init__(self, model_dir: str):
        vocab_path = os.path.join(model_dir, "vocab.json")
        merges_path = os.path.join(model_dir, "merges.txt")
        tokenizer_json_path = os.path.join(model_dir, "tokenizer.json")

        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab: Dict[str, int] = json.load(f)
            merges = self._load_merges_txt(merges_path)
        elif os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tok = json.load(f)
            self.vocab = tok["model"]["vocab"]
            merges = tok["model"]["merges"]
            merges = [tuple(m) if isinstance(m, list) else tuple(m.split()) for m in merges]
        else:
            raise FileNotFoundError(
                "Need vocab.json + merges.txt, or tokenizer.json, in the model directory."
            )

        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {tuple(m): i for i, m in enumerate(merges)}
        self.cache: Dict[str, List[str]] = {}

        self.byte_encoder, self.byte_decoder = bytes_to_unicode()

        self.special_tokens = self._load_special_tokens(model_dir)
        self.special_tokens = {tok for tok in self.special_tokens if tok in self.vocab}
        self.special_pattern: Optional[re.Pattern[str]]
        if self.special_tokens:
            pieces = sorted((re.escape(s) for s in self.special_tokens), key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(pieces) + ")")
        else:
            self.special_pattern = None

        self.eos_token = "<|im_end|>"
        self.pad_token = "<|endoftext|>"
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"

        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab.get(self.pad_token, self.eos_token_id)

    @staticmethod
    def _load_merges_txt(path: str) -> List[Tuple[str, str]]:
        merges: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return merges

    def _load_special_tokens(self, model_dir: str) -> set[str]:
        special = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}
        cfg_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
                val = cfg.get(key)
                if isinstance(val, str):
                    special.add(val)
                elif isinstance(val, dict) and isinstance(val.get("content"), str):
                    special.add(val["content"])

            for tok in cfg.get("additional_special_tokens", []) or []:
                if isinstance(tok, str):
                    special.add(tok)
                elif isinstance(tok, dict) and isinstance(tok.get("content"), str):
                    special.add(tok["content"])

            dec = cfg.get("added_tokens_decoder", {}) or {}
            for entry in dec.values():
                if isinstance(entry, dict) and entry.get("special") and isinstance(entry.get("content"), str):
                    special.add(entry["content"])

        return special

    def bpe(self, token: str) -> List[str]:
        cached = self.cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if len(word) == 1:
            self.cache[token] = [token]
            return [token]

        while True:
            pairs = get_pairs(word)
            if not pairs:
                break
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

        out = list(word)
        self.cache[token] = out
        return out

    def _encode_normal_text(self, text: str) -> List[int]:
        text = unicodedata.normalize("NFC", text)
        ids: List[int] = []

        for piece in qwen_pretokenize(text):
            byte_level = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for bpe_piece in self.bpe(byte_level):
                try:
                    ids.append(self.vocab[bpe_piece])
                except KeyError as exc:
                    raise KeyError(f"Tokenizer piece not found in vocab: {bpe_piece!r}") from exc

        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        if self.special_pattern is None:
            return self._encode_normal_text(text)

        ids: List[int] = []
        pos = 0
        for match in self.special_pattern.finditer(text):
            if match.start() > pos:
                ids.extend(self._encode_normal_text(text[pos : match.start()]))
            special = match.group(0)
            ids.append(self.vocab[special])
            pos = match.end()

        if pos < len(text):
            ids.extend(self._encode_normal_text(text[pos:]))

        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        byte_buffer = bytearray()
        text_parts: List[str] = []

        def flush_bytes() -> None:
            nonlocal byte_buffer
            if byte_buffer:
                text_parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer = bytearray()

        for idx in ids:
            token = self.id_to_token.get(int(idx), "")
            if skip_special_tokens and token in self.special_tokens:
                continue

            if token in self.special_tokens:
                flush_bytes()
                text_parts.append(token)
                continue

            for ch in token:
                b = self.byte_decoder.get(ch)
                if b is None:
                    flush_bytes()
                    text_parts.append(ch)
                else:
                    byte_buffer.append(b)

        flush_bytes()
        return "".join(text_parts)

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        if not messages or messages[0].get("role") != "system":
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                }
            ] + messages

        chunks: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            chunks.append(f"{self.im_start}{role}\n{content}{self.im_end}\n")

        if add_generation_prompt:
            chunks.append(f"{self.im_start}assistant\n")

        return "".join(chunks)


# -----------------------------
# Qwen2 model, pure PyTorch
# -----------------------------

@dataclass
class Qwen2ConfigPure:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    eos_token_id: int
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, path: str) -> "Qwen2ConfigPure":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
            eos_token_id=int(cfg.get("eos_token_id", 151645)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        )


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(original_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q: [B, QH, T, D], k: [B, KVH, T, D]
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq.float())
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=q.dtype).unsqueeze(1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [B, KVH, T, D] -> [B, QH, T, D]
    if n_rep == 1:
        return x
    bsz, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: Qwen2ConfigPure, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2ConfigPure, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape

        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, position_ids, self.inv_freq)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_kv = (k, v) if use_cache else None

        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        attn_scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scaling

        key_len = k_rep.shape[2]
        if q_len > 1:
            query_positions = torch.arange(past_len, past_len + q_len, device=x.device)[:, None]
            key_positions = torch.arange(0, key_len, device=x.device)[None, :]
            causal_mask = key_positions > query_positions
            attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(attn_scores.dtype).min)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=q.dtype)
        attn_out = torch.matmul(attn_probs, v_rep)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_out), new_kv


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2ConfigPure, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, device=device, dtype=dtype)
        self.mlp = Qwen2MLP(cfg, device=device, dtype=dtype)
        self.input_layernorm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x),
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = residual + attn_out

        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen2ForCausalLMPure(nn.Module):
    def __init__(self, cfg: Qwen2ConfigPure, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if not cfg.tie_word_embeddings:
            raise ValueError("This minimal implementation expects tied word embeddings.")

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)  # type: ignore[list-item]
            past_len = 0
        else:
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)

        x = self.embed_tokens(input_ids)
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer, layer_past in zip(self.layers, past_key_values):
            x, layer_cache = layer(x, position_ids=position_ids, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                assert layer_cache is not None
                new_cache.append(layer_cache)

        x = self.norm(x)

        # Tied output projection: logits = hidden @ embedding.T
        logits = F.linear(x, self.embed_tokens.weight)
        return logits, new_cache if use_cache else None


# -----------------------------
# Generation
# -----------------------------

def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    name = name.lower()
    if name == "auto":
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    # logits: [1, vocab]
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_p < 1.0:
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, next_sorted).squeeze(-1)

    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: Qwen2ForCausalLMPure,
    tokenizer: Qwen2TokenizerPure,
    messages: List[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache = model(input_ids, past_key_values=None, use_cache=True)

    generated: List[int] = []
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_token = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        token_id = int(next_token.item())

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        step_ids = next_token.view(1, 1).to(device=device)
        logits, cache = model(step_ids, past_key_values=cache, use_cache=True)
        next_logits = logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_model(model_dir: str, device: torch.device, dtype: torch.dtype) -> Qwen2ForCausalLMPure:
    cfg = Qwen2ConfigPure.from_json(os.path.join(model_dir, "config.json"))
    model = Qwen2ForCausalLMPure(cfg, device=device, dtype=dtype)
    load_safetensors_into_model(model, model_dir)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory containing Qwen2.5-0.5B-Instruct files.")
    parser.add_argument("--prompt", default=None, help="Single prompt. If omitted, starts an interactive loop.")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="System prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype, device)

    tokenizer = Qwen2TokenizerPure(args.model_dir)
    model = build_model(args.model_dir, device=device, dtype=dtype)

    if args.prompt is not None:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        print(
            generate(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
        )
        return

    history: List[dict] = [{"role": "system", "content": args.system}]
    print("Pure PyTorch Qwen2.5-0.5B-Instruct chat. Type 'exit' or 'quit' to stop.")

    while True:
        user_text = input("\nUser: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        answer = generate(
            model=model,
            tokenizer=tokenizer,
            messages=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(f"\nQwen: {answer}")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
