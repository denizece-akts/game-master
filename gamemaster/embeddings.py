import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .config import CONFIG, OUTPUT_DIR, DEVICE
from .utils import clamp_sentences, clamp_chars, normalize_whitespace, _json_safe
from .data import load_data


def emb_encode(emb_model, texts, use_normalize: bool, bsz=64):
    outs = []
    for i in range(0, len(texts), bsz):
        sub = texts[i : i + bsz]
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

    print("Loading embedding model (from HF cache):", CONFIG["embedding_model"])
    emb_model = SentenceTransformer(CONFIG["embedding_model"], device=DEVICE)
    emb_model.max_seq_length = 512
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
    R = emb_encode(emb_model, review_texts, use_normalize, bsz=CONFIG["embedding_batch_size"])
    rev_ids = np.arange(len(R), dtype=np.int64)
    rev_index.add_with_ids(R, rev_ids)
    faiss.write_index(rev_index, str(rev_faiss_path))
    print(f"✅ Saved {rev_faiss_path.name} with {len(R)} vectors.")

    games_unique = games_df.drop_duplicates(subset=["_game_key"]).reset_index(drop=True)
    game_texts = make_game_texts(games_unique)
    G = emb_encode(emb_model, game_texts, use_normalize, bsz=CONFIG["embedding_batch_size"])
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
            {"num_games": int(len(games_unique)), "faiss_to_rowpos": list(range(len(games_unique)))},
            f,
        )

    with open(rev_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe(
                {
                    "config": CONFIG,
                    "desc_csv_path_used": str(desc_path),
                    "reviews_csv_path_used": str(reviews_path),
                    "num_games_rows": int(len(games_df)),
                    "num_reviews_rows": int(len(reviews_df)),
                    "embedding_model": CONFIG["embedding_model"],
                    "llm_model": CONFIG["llm_model"],
                }
            ),
            f,
            indent=2,
        )

    print("✅ Embeddings done & saved.")
    return emb_model, use_normalize, game_index, rev_index, games_unique, reviews_df
