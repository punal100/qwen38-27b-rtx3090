"""Build a vocab-truncated draft head for MTP speculative decoding.

The MTP drafter has to run the 248k-row lm_head once per draft token, and at
4-6 drafts per step that head read (1.3 GB int8) dominates the draft cost.
Speculative decoding stays exact no matter what the drafter proposes, so the
drafter can use a head restricted to the N most frequent tokens: tokens
outside the shortlist are simply never drafted (that position gets rejected
and the target's own sample is used, as always).

This script counts token frequencies over a text corpus (Danish + English +
code + the model's own outputs), picks the top N ids (plus special tokens),
slices those rows out of the already-int8-quantized lm_head, and stores them
as `mtp.draft_lm_head.*` in model_extra_tensors.safetensors, plus the id map in
`mtp_draft_vocab_ids.pt`. Needs the matching vLLM patch
(patches/qwen3_5-mtp-draft-vocab.patch) to be used.

Usage (from the repo root):
  python prepare/build_draft_vocab.py /path/to/model --ids prepare/draft_vocab_ids.json  # shipped id list
  python prepare/build_draft_vocab.py /path/to/model --n 40960 --corpus f1 f2 ...        # or count your own
Corpus files: .txt/.jsonl (uses "prompt"/"response"/"messages"/"text" fields)/.parquet(text)/.py
The shipped draft_vocab_ids.json was counted over Danish web text (fineweb-2),
English Wikipedia, Python source and the model's own chat outputs (8.8M tokens);
held-out coverage 95%.
"""
import glob, json, os, sys, shutil, collections, uuid
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def fsync_dir(path):
    fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj):
    tmp = path + f".tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def atomic_torch(path, obj):
    tmp = path + f".tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "xb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def ensure_backup(src, backup):
    if os.path.lexists(backup):
        if os.path.islink(backup) or not os.path.isfile(backup):
            raise RuntimeError(f"unsafe draft backup path: {backup}")
        return
    tmp = backup + f".tmp-{uuid.uuid4().hex}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(src, "rb") as source, os.fdopen(fd, "wb") as dest:
            shutil.copyfileobj(source, dest)
            dest.flush()
            os.fsync(dest.fileno())
        try:
            os.link(tmp, backup)
        except FileExistsError:
            if os.path.islink(backup) or not os.path.isfile(backup):
                raise RuntimeError(f"unsafe draft backup path: {backup}")
        fsync_dir(backup)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass

d = sys.argv[1].rstrip("/") + "/"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40960
corpus = sys.argv[sys.argv.index("--corpus") + 1:] if "--corpus" in sys.argv else []
ids_file = sys.argv[sys.argv.index("--ids") + 1] if "--ids" in sys.argv else None

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d)

def texts_from(path, limit_bytes=20_000_000):
    n = 0
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            yield t; n += len(t)
            if n > limit_bytes: return
    elif path.endswith(".jsonl"):
        for line in open(path):
            try: r = json.loads(line)
            except Exception: continue
            parts = []
            for k in ("prompt", "response", "text"):
                if isinstance(r.get(k), str): parts.append(r[k])
            if isinstance(r.get("messages"), list):
                parts += [m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str)]
            t = "\n".join(parts); yield t; n += len(t)
            if n > limit_bytes: return
    else:
        t = open(path, errors="ignore").read(); yield t

counts = collections.Counter()
held = collections.Counter()
total = 0
if ids_file:
    ids = sorted(set(json.load(open(ids_file))))
    print(f"using {len(ids)} ids from {ids_file}")
    corpus = []
for i, path in enumerate(corpus):
    for j, t in enumerate(texts_from(path)):
        ids = tok(t, add_special_tokens=False).input_ids
        (held if j % 10 == 0 else counts).update(ids)
        total += len(ids)
print(f"corpus tokens: {total}")

special = set(tok.all_special_ids)
if ids_file:
    special = set()
for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0: special.add(tid)
if not ids_file:
    top = [t for t, _ in counts.most_common() if t not in special][: N - len(special)]
    ids = sorted(set(top) | special)
    cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
    print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
    for n_try in (16384, 32768, 49152, 65536):
        s = set(t for t, _ in counts.most_common(n_try)) | special
        c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
        print(f"  coverage at N={n_try}: {c*100:.2f}%")
    atomic_json(d + "draft_vocab_ids.json", ids)
    print(f"id list written to {d}draft_vocab_ids.json (copy it next to this script to reuse)")

# slice lm_head rows
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
head_shard = wm["lm_head.weight_packed"]
with safe_open(d + head_shard, framework="pt") as f:
    wp = f.get_tensor("lm_head.weight_packed")   # [vocab, K/8] int32
    ws = f.get_tensor("lm_head.weight_scale")    # [vocab, K/group]
    shape = f.get_tensor("lm_head.weight_shape")
ids_t = torch.tensor(ids, dtype=torch.int64)
sub_p = wp.index_select(0, ids_t).contiguous()
sub_s = ws.index_select(0, ids_t).contiguous()
sub_shape = torch.tensor([len(ids), int(shape[1])], dtype=torch.int64)
print(f"draft head: packed {tuple(sub_p.shape)} {sub_p.dtype}, scales {tuple(sub_s.shape)} {sub_s.dtype}, "
      f"{(sub_p.numel()*4 + sub_s.numel()*2)/1e6:.0f} MB")

extra = "model_extra_tensors.safetensors"
# The three quant_*.py scripts leave the base model's extras in this file; a
# single-shard export processed by prepare/quant_heads_stream.py has no extras
# file at all (#37) -- the draft head becomes its first content.
tensors = {}
meta = None
extra_path = d + extra
draft_keys = {
    "mtp.draft_lm_head.weight_packed",
    "mtp.draft_lm_head.weight_scale",
    "mtp.draft_lm_head.weight_shape",
}
expected_mtp = {
    key for key, shard in wm.items()
    if shard == extra and key.startswith("mtp.") and key not in draft_keys
}
indexed_extra = {key for key, shard in wm.items() if shard == extra}
indexed_draft = indexed_extra & draft_keys
if indexed_draft not in (set(), draft_keys):
    raise RuntimeError(f"index has incomplete draft-head inventory: {sorted(indexed_draft)}")
existing_keys = set()
if os.path.exists(extra_path):
    with safe_open(extra_path, framework="pt") as f:
        meta = f.metadata()
        existing_keys = set(f.keys())
        missing = expected_mtp - existing_keys
        if missing:
            raise RuntimeError(f"existing extra shard is missing indexed MTP tensors: {sorted(missing)}")
        allowed = {frozenset(indexed_extra)}
        if not indexed_draft:
            allowed.add(frozenset(indexed_extra | draft_keys))
        if frozenset(existing_keys) not in allowed:
            raise RuntimeError("existing extra shard does not match index inventory")
        for k in existing_keys:
            tensors[k] = f.get_tensor(k)
    backup_path = extra_path + ".bak-draft"
    expected_backup = indexed_extra - draft_keys
    if expected_backup:
        ensure_backup(extra_path, backup_path)
        try:
            with safe_open(backup_path, framework="pt") as f:
                backup_keys = set(f.keys())
        except Exception as exc:
            raise RuntimeError(f"draft backup is unreadable: {backup_path}") from exc
        if backup_keys != expected_backup:
            raise RuntimeError(f"draft backup inventory mismatch: {backup_path}")
    elif os.path.lexists(backup_path):
        if os.path.islink(backup_path) or not os.path.isfile(backup_path):
            raise RuntimeError(f"unsafe draft backup path: {backup_path}")
        try:
            with safe_open(backup_path, framework="pt") as f:
                if set(f.keys()) != draft_keys:
                    raise RuntimeError(f"draft backup inventory mismatch: {backup_path}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"draft backup is unreadable: {backup_path}") from exc
elif expected_mtp:
    raise RuntimeError(f"extra shard is absent but index expects MTP tensors: {sorted(expected_mtp)}")
tensors["mtp.draft_lm_head.weight_packed"] = sub_p
tensors["mtp.draft_lm_head.weight_scale"] = sub_s
tensors["mtp.draft_lm_head.weight_shape"] = sub_shape
tmp = extra_path + f".tmp-{uuid.uuid4().hex}"
try:
    save_file(tensors, tmp, metadata=meta or {"format": "pt"})
    with safe_open(tmp, framework="pt") as f:
        written_keys = set(f.keys())
    required_keys = existing_keys | draft_keys
    if written_keys != required_keys or not expected_mtp <= written_keys:
        raise RuntimeError(
            "draft extra shard inventory mismatch: "
            f"missing={sorted(required_keys - written_keys)} extra={sorted(written_keys - required_keys)}"
        )
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, extra_path)
    fsync_dir(extra_path)
finally:
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
for s in ("weight_packed", "weight_scale", "weight_shape"):
    wm[f"mtp.draft_lm_head.{s}"] = extra
atomic_torch(d + "mtp_draft_vocab_ids.pt", ids_t)
atomic_json(d + "model.safetensors.index.json", idx)
print("done")
