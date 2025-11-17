import os
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from .config import CONFIG, OUTPUT_DIR, DEVICE, HF_TOKEN
from .utils import (
    clamp_sentences,
    clamp_chars,
    normalize_whitespace,
    _json_safe,
    sha256_file,
)
from .data import load_data, safe_read_csv


def _ensure_local_embedding_model() -> str:
    remote_id = CONFIG["embedding_model"]
    local_dir = Path(CONFIG.get("embedding_local_dir", "./bge_model"))

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Using existing embedding model dir: {local_dir}")
        return str(local_dir)

    print(f"Embedding model dir {local_dir} missing/empty, downloading from {remote_id}...")
    local_dir.mkdir(parents=True, exist_ok=True)

    token = HF_TOKEN or os.environ.get("HF_TOKEN", None)

    snapshot_download(
        repo_id=remote_id,
        token=token,
        local_dir=str(local_dir),
    )

    print("✅ Embedding model download complete.")
    return str(local_dir)


def _load_embedding_model() -> SentenceTransformer:
    local_dir = _ensure_local_embedding_model()
    print(f"Loading embedding model from: {local_dir}")
    emb_model = SentenceTransformer(local_dir, device=DEVICE)
    emb_model.max_seq_length = 512
    return emb_model


def emb_encode(emb_model: SentenceTransformer, texts, use_normalize: bool, bsz: int = 64):
    outs = []
    for i in range(0, len(texts), bsz):
        sub = texts[i : i + bsz]
        if not sub:
            continue
        with torch.inference_mode():
            v = emb_model.encode(
                sub,
                convert_to_numpy=True,
                normalize_embeddings=use_normalize,
                batch_size=min(bsz, len(sub)),
                show_progress_bar=False,
            ).astype(np.float32)
        outs.append(v)
    if outs:
        return np.vstack(outs)
    return np.zeros((0, emb_model.get_sentence_embedding_dimension()), dtype=np.float32)


def make_review_texts(df_slice: pd.DataFrame):
    out = []
    for _, row in df_slice.iterrows():
        parts = []
        parts.append(f"Game: {row.get('game_name', '')}")
        rv = row.get("review", "")
        parts.append(f"Review: {clamp_sentences(rv, 3)}" if pd.notna(rv) else "")
        rcmd = row.get("recommendation", "")
        parts.append(f"Recommendation: {rcmd}" if pd.notna(rcmd) else "")
        hp = row.get("hours_played", np.nan)
        parts.append(f"Hours: {int(hp)}" if pd.notna(hp) else "")
        hl = row.get("helpful", np.nan)
        parts.append(f"Helpful: {int(hl)}" if pd.notna(hl) else "")
        parts = [p for p in parts if p]
        out.append("\n".join(parts))
    return out


def make_game_texts(df_slice: pd.DataFrame):
    out = []
    for _, g in df_slice.iterrows():
        parts = [
            f"Name: {g.get('name', '')}",
            f"Genres: {g.get('genres_str', '')}",
            f"Developer: {g.get('developer', '')}",
            f"Publisher: {g.get('publisher', '')}",
            f"Rating: {g.get('overall_player_rating', '')}",
            f"ShortDesc: {clamp_sentences(g.get('short_description', '') or '', 2)}",
        ]
        min_req = normalize_whitespace(g.get("minimum_system_requirement", "") or "")
        rec_req = normalize_whitespace(g.get("recommend_system_requirement", "") or "")
        if min_req:
            parts.append(f"MinReq: {clamp_chars(min_req, 300)}")
        if rec_req:
            parts.append(f"RecReq: {clamp_chars(rec_req, 300)}")
        out.append("\n".join(parts))
    return out


def build_indices():
    games_df, reviews_df, desc_path, reviews_path = load_data()

    print("Loading embedding model:", CONFIG["embedding_model"])
    emb_model = _load_embedding_model()
    use_normalize = CONFIG["normalize_embeddings"]

    rev_faiss_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_faiss.index"
    game_faiss_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_games_faiss.index"
    games_unique_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_games_unique.parquet"
    game_idmap_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_games_idmap.json"
    rev_meta_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_meta.json"

    dim = emb_model.get_sentence_embedding_dimension()
    rev_base = faiss.IndexFlatIP(dim) if use_normalize else faiss.IndexFlatL2(dim)
    rev_index = faiss.IndexIDMap2(rev_base)
    game_base = faiss.IndexFlatIP(dim) if use_normalize else faiss.IndexFlatL2(dim)
    game_index = faiss.IndexIDMap2(game_base)

    review_texts = make_review_texts(reviews_df)
    R = emb_encode(
        emb_model,
        review_texts,
        use_normalize,
        bsz=CONFIG["embedding_batch_size"],
    )
    rev_ids = np.arange(len(R), dtype=np.int64)
    rev_index.add_with_ids(R, rev_ids)
    faiss.write_index(rev_index, str(rev_faiss_path))
    print(f"✅ Saved {rev_faiss_path.name} with {len(R)} vectors.")

    games_unique = games_df.drop_duplicates(subset=["_game_key"]).reset_index(drop=True)
    game_texts = make_game_texts(games_unique)
    G = emb_encode(
        emb_model,
        game_texts,
        use_normalize,
        bsz=CONFIG["embedding_batch_size"],
    )
    game_ids = np.arange(len(G), dtype=np.int64)
    game_index.add_with_ids(G, game_ids)
    faiss.write_index(game_index, str(game_faiss_path))
    print(f"✅ Saved {game_faiss_path.name} with {len(G)} vectors.")

    try:
        games_unique.to_parquet(games_unique_path, index=False)
    except Exception:
        csv_fallback = games_unique_path.with_suffix(".csv")
        games_unique.to_csv(csv_fallback, index=False)
        games_unique_path = csv_fallback

    with open(game_idmap_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_games": int(len(games_unique)),
                "faiss_to_rowpos": list(range(len(games_unique))),
            },
            f,
        )

    desc_full = Path(CONFIG["desc_csv_path_full"])
    reviews_full = Path(CONFIG["reviews_csv_path_full"])
    desc_checksum = sha256_file(desc_full)
    reviews_checksum = sha256_file(reviews_full)

    meta = {
        "config_subset": {
            "embedding_model": CONFIG["embedding_model"],
            "normalize_embeddings": CONFIG["normalize_embeddings"],
            "topN_for_subset": CONFIG["topN_for_subset"],
        },
        "desc_csv_path_used": str(desc_path),
        "reviews_csv_path_used": str(reviews_path),
        "source_paths": {
            "desc_csv_full": str(desc_full),
            "reviews_csv_full": str(reviews_full),
        },
        "source_checksums": {
            "desc_csv_full": desc_checksum,
            "reviews_csv_full": reviews_checksum,
        },
        "num_games_rows": int(len(games_df)),
        "num_reviews_rows": int(len(reviews_df)),
        "embedding_model": CONFIG["embedding_model"],
        "llm_model": CONFIG["llm_model"],
    }

    with open(rev_meta_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2)

    print("✅ Embeddings done & saved.")
    return emb_model, use_normalize, game_index, rev_index, games_unique, reviews_df


def load_or_build_indices():
    rev_faiss_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_faiss.index"
    game_faiss_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_games_faiss.index"
    games_unique_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_games_unique.parquet"
    rev_meta_path = OUTPUT_DIR / f"{CONFIG['artifact_prefix']}_meta.json"

    games_unique_csv_path = games_unique_path.with_suffix(".csv")

    artifacts_exist = (
        rev_faiss_path.exists()
        and game_faiss_path.exists()
        and rev_meta_path.exists()
        and (games_unique_path.exists() or games_unique_csv_path.exists())
    )

    if not artifacts_exist:
        print("Artifacts missing, rebuilding indices...")
        return build_indices()

    try:
        with open(rev_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        print("Meta file unreadable, rebuilding indices...")
        return build_indices()

    try:
        source_paths = meta["source_paths"]
        source_checksums = meta["source_checksums"]
        cfg_sub = meta.get("config_subset", {})
    except KeyError:
        print("Meta file incomplete, rebuilding indices...")
        return build_indices()

    desc_full_cfg = CONFIG["desc_csv_path_full"]
    reviews_full_cfg = CONFIG["reviews_csv_path_full"]

    if (
        source_paths.get("desc_csv_full") != desc_full_cfg
        or source_paths.get("reviews_csv_full") != reviews_full_cfg
    ):
        print("Source CSV paths changed, rebuilding indices...")
        return build_indices()

    desc_full = Path(desc_full_cfg)
    reviews_full = Path(reviews_full_cfg)
    current_desc_checksum = sha256_file(desc_full)
    current_reviews_checksum = sha256_file(reviews_full)

    if (
        current_desc_checksum != source_checksums.get("desc_csv_full")
        or current_reviews_checksum != source_checksums.get("reviews_csv_full")
    ):
        print("Source CSV content changed, rebuilding indices...")
        return build_indices()

    if (
        cfg_sub.get("embedding_model") != CONFIG["embedding_model"]
        or cfg_sub.get("normalize_embeddings") != CONFIG["normalize_embeddings"]
        or cfg_sub.get("topN_for_subset") != CONFIG["topN_for_subset"]
    ):
        print("Config affecting indices changed, rebuilding indices...")
        return build_indices()

    print("✅ Existing indices are up to date; loading from disk.")

    rev_index = faiss.read_index(str(rev_faiss_path))
    game_index = faiss.read_index(str(game_faiss_path))

    if games_unique_path.exists():
        games_unique = pd.read_parquet(games_unique_path)
    else:
        games_unique = pd.read_csv(games_unique_csv_path)

    reviews_subset_path = Path(meta["reviews_csv_path_used"])
    reviews_df = safe_read_csv(reviews_subset_path)

    emb_model = _load_embedding_model()
    use_normalize = CONFIG["normalize_embeddings"]

    return emb_model, use_normalize, game_index, rev_index, games_unique, reviews_df

