"""FP8-dynamic (W8A8) quantizer for Hugging Face safetensors models.

Writes a compressed-tensors checkpoint that vLLM loads directly:
  * Linear weights -> float8_e4m3fn with one float32 scale per output row
  * activations are quantized per token at run time (dynamic)
  * embeddings, lm_head, norms, routers and vision/audio parts stay as they are

Streams tensor by tensor (a few hundred MB of RAM), needs only numpy.
Usage: python fp8_dynamic_quant.py SRC_DIR DST_DIR [--ignore REGEX ...] [--shard-gb 4]
Progress lines: "PROGRESS <done_bytes> <total_bytes>", other lines are messages.
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

FP8_MAX = 448.0
DEFAULT_IGNORE = [r".*lm_head.*", r".*embed.*", r".*vision.*", r".*visual.*", r".*audio.*",
                  r".*multi_modal_projector.*", r".*mm_proj.*", r".*siglip.*", r".*router.*",
                  r".*\.gate$", r".*shared_expert_gate.*"]
SRC_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
         "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "U32": 4, "U64": 8}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def to_f32(raw, dtype):
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def fp8_e4m3fn_encode(x):
    """float32 array (|x| <= 448) -> uint8 float8_e4m3fn codes, round to nearest even."""
    a = np.abs(x).astype(np.float32)
    _, e = np.frexp(a)                                    # a = m * 2**e, m in [0.5, 1)
    step = np.where(a >= 2.0 ** -6, np.ldexp(np.float32(1), e - 4), np.float32(2.0 ** -9))
    q = np.minimum(np.rint(a / step) * step, FP8_MAX).astype(np.float32)
    m2, e2 = np.frexp(q)
    normal = q >= 2.0 ** -6
    expf = np.where(normal, e2 + 6, 0)                    # biased exponent (bias 7): (e2 - 1) + 7
    mant = np.where(normal, np.rint((m2 * 2 - 1) * 8), np.rint(q / 2.0 ** -9))
    code = (expf.astype(np.uint32) << 3) | mant.astype(np.uint32)
    code = np.where(q == 0, 0, code)
    code |= (np.signbit(x).astype(np.uint32) << 7)
    return code.astype(np.uint8)


def fp8_e4m3fn_decode(codes):
    c = codes.astype(np.int32)
    s = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 0x7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    return (s * v).astype(np.float32)


def _fast_encoder():
    """ml_dtypes' C conversion (much faster), used only if it matches the reference exactly."""
    try:
        import ml_dtypes
        f8 = ml_dtypes.float8_e4m3fn
        t = np.random.default_rng(1).normal(0, 60, 200000).astype(np.float32).clip(-FP8_MAX, FP8_MAX)
        t[:4] = [0.0, -0.0, FP8_MAX, -FP8_MAX]
        if np.array_equal(t.astype(f8).view(np.uint8), fp8_e4m3fn_encode(t)):
            return lambda q: q.astype(f8).view(np.uint8)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--shard-gb", type=float, default=4.0)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    ignore = DEFAULT_IGNORE + [p for p in args.ignore if p.strip()]
    ign_re = [re.compile(p) for p in ignore]

    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("quantization_config") or any(isinstance(cfg.get(k), dict) and cfg[k].get("quantization_config")
                                             for k in ("text_config", "llm_config")):
        print("ERROR: this model is already quantized")
        return 2
    files = sorted(src.glob("*.safetensors"))
    if not files:
        print("ERROR: no .safetensors files in the source folder")
        return 2

    # ---- plan: every source tensor -> output tensors with known sizes ----
    plan, total_src = [], 0
    for fp in files:
        hdr, data_start = read_header(fp)
        for name, meta in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = meta["data_offsets"]
            dt, shape = meta["dtype"], meta["shape"]
            total_src += b - a
            module = name[:-len(".weight")] if name.endswith(".weight") else None
            if dt.startswith("F8"):
                print(f"ERROR: {name} is already FP8 - this model is already quantized")
                return 2
            if "expert" in name and len(shape) == 3:
                print(f"ERROR: {name} is a fused 3-D expert tensor; this tool only handles "
                      "per-expert 2-D weights")
                return 2
            quant = (module is not None and len(shape) == 2 and dt in SRC_DTYPES
                     and not any(r.match(module) for r in ign_re))
            plan.append({"file": fp, "start": data_start + a, "nbytes": b - a, "name": name,
                         "dtype": dt, "shape": shape, "quant": quant, "module": module})
    n_q = sum(1 for t in plan if t["quant"])
    if not n_q:
        print("ERROR: no Linear weights found to quantize")
        return 2

    outs = []        # (plan item, out name, dtype, shape, nbytes, kind)
    for t in plan:
        if t["quant"]:
            rows, cols = t["shape"]
            outs.append((t, t["name"], "F8_E4M3", [rows, cols], rows * cols, "q"))
            outs.append((t, t["module"] + ".weight_scale", "F32", [rows, 1], rows * 4, "s"))
        else:
            outs.append((t, t["name"], t["dtype"], t["shape"], t["nbytes"], "copy"))
    limit = int(args.shard_gb * 1024 ** 3)
    shards, cur, cur_size = [], [], 0
    for o in outs:
        if cur and cur_size + o[4] > limit and o[5] != "s":      # keep a weight next to its scale
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(o)
        cur_size += o[4]
    if cur:
        shards.append(cur)
    out_total = sum(o[4] for o in outs)
    print(f"{n_q} Linear weights to FP8, {len(plan) - n_q} tensors kept; "
          f"{total_src / 1024 ** 3:.2f} GiB -> {out_total / 1024 ** 3:.2f} GiB in {len(shards)} file(s)")
    free = shutil.disk_usage(dst.parent if dst.parent.exists() else dst.anchor).free
    if free < out_total + 256 * 1024 ** 2:
        print(f"ERROR: not enough disk space: need {out_total / 1024 ** 3:.1f} GiB, "
              f"{free / 1024 ** 3:.1f} GiB free")
        return 2

    encode = _fast_encoder() or fp8_e4m3fn_encode
    print("FP8 conversion: " + ("ml_dtypes (fast)" if encode is not fp8_e4m3fn_encode else "numpy"), flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.time()
    scales = {}
    weight_map = {}
    nshards = len(shards)
    for si, shard in enumerate(shards, 1):
        fname = f"model-{si:05d}-of-{nshards:05d}.safetensors"
        header, off = {}, 0
        for (_t, name, dt, shape, nb, _k) in shard:
            header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
            off += nb
            weight_map[name] = fname
        header["__metadata__"] = {"format": "pt"}
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(dst / fname, "wb") as out:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            for (t, name, dt, shape, nb, kind) in shard:
                if kind == "s":
                    out.write(scales.pop(t["name"]).astype("<f4").tobytes())
                    continue
                with open(t["file"], "rb") as f:
                    f.seek(t["start"])
                    if kind == "copy":
                        left = t["nbytes"]
                        while left:
                            buf = f.read(min(left, 64 * 1024 ** 2))
                            if not buf:
                                raise IOError(f"unexpected end of file reading {t['name']}")
                            out.write(buf)
                            left -= len(buf)
                            done += len(buf)
                            print(f"PROGRESS {done} {total_src}", flush=True)
                        continue
                    rows, cols = t["shape"]
                    es = SRC_DTYPES[t["dtype"]]
                    step = max(1, (64 * 1024 ** 2) // max(1, cols * es))
                    sc_all = np.empty(rows, dtype=np.float32)
                    for r0 in range(0, rows, step):
                        r1 = min(rows, r0 + step)
                        raw = f.read((r1 - r0) * cols * es)
                        w = to_f32(raw, t["dtype"]).reshape(r1 - r0, cols)
                        amax = np.abs(w).max(axis=1)
                        sc = np.maximum(amax / FP8_MAX, np.float32(1e-12)).astype(np.float32)
                        q = np.clip(w / sc[:, None], -FP8_MAX, FP8_MAX)
                        out.write(encode(q).tobytes())
                        sc_all[r0:r1] = sc
                        done += len(raw)
                        print(f"PROGRESS {done} {total_src}", flush=True)
                    scales[t["name"]] = sc_all
        print(f"wrote {fname}", flush=True)

    total_out = sum(o[4] for o in outs)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_out}, "weight_map": weight_map}, indent=2))
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
            shutil.copyfile(p, dst / p.name)           # copyfile: no chmod (NTFS-safe)
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True,
                        "dynamic": False, "group_size": None, "block_structure": None, "observer": "minmax"},
            "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True,
                                  "dynamic": True, "group_size": None, "block_structure": None, "observer": None},
            "output_activations": None,
            "format": "float-quantized"}},
        "ignore": ["re:" + p for p in ignore],
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "sparsity_config": {},
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"DONE in {(time.time() - t0) / 60:.1f} min: {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
