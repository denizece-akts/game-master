from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, OUTPUT_DIR
from .utils import (
    fix_mojibake,
    normalize_whitespace,
    parse_listlike,
    parse_date_any,
    to_number,
    make_game_key,
)


def build_top10_subsets(desc_csv, reviews_csv, out_dir: Path, topN: int = 10):
    desc = pd.read_csv(desc_csv)
    rev = pd.read_csv(reviews_csv)

    rev = rev.reset_index().rename(columns={"index": "original_index_reviews"})
    desc = desc.reset_index().rename(columns={"index": "original_index_desc"})

    rev["_game_key"] = rev["game_name"].map(make_game_key)
    desc["_game_key"] = desc["name"].map(make_game_key)

    top_keys = (
        rev.groupby("_game_key")
        .size()
        .sort_values(ascending=False)
        .head(topN)
        .index
        .tolist()
    )

    rev_top = rev[rev["_game_key"].isin(top_keys)].copy()
    desc_top = desc[desc["_game_key"].isin(top_keys)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    rev_top.to_csv(out_dir / "steam_game_reviews__TOP10_GAMES.csv", index=False)
    desc_top.to_csv(out_dir / "games_description__TOP10_GAMES.csv", index=False)

    rev_top[["original_index_reviews"]].to_csv(
        out_dir / "steam_game_reviews__TOP10_GAMES_indices.csv", index=False
    )
    desc_top[["original_index_desc"]].to_csv(
        out_dir / "games_description__TOP10_GAMES_indices.csv", index=False
    )

    print("✅ Built TOP-10 subsets.")
    return (
        out_dir / "games_description__TOP10_GAMES.csv",
        out_dir / "steam_game_reviews__TOP10_GAMES.csv",
    )


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quoting=0,
            encoding="utf-8",
            errors="replace",
        )


def load_data():
    desc_p, rev_p = build_top10_subsets(
        CONFIG["desc_csv_path_full"],
        CONFIG["reviews_csv_path_full"],
        OUTPUT_DIR,
        CONFIG["topN_for_subset"],
    )

    assert Path(desc_p).exists(), f"Not found: {desc_p}"
    assert Path(rev_p).exists(), f"Not found: {rev_p}"

    games_df = safe_read_csv(desc_p)
    reviews_df = safe_read_csv(rev_p)

    print("Loaded:", desc_p, games_df.shape)
    print("Loaded:", rev_p, reviews_df.shape)

    desc_cols_keep = [
        "name",
        "short_description",
        "long_description",
        "genres",
        "minimum_system_requirement",
        "recommend_system_requirement",
        "release_date",
        "developer",
        "publisher",
        "overall_player_rating",
    ]
    desc_cols_keep = [c for c in desc_cols_keep if c in games_df.columns]
    games_df = games_df[desc_cols_keep].copy()

    for c in [
        "name",
        "short_description",
        "long_description",
        "overall_player_rating",
        "developer",
        "publisher",
        "minimum_system_requirement",
        "recommend_system_requirement",
    ]:
        if c in games_df.columns:
            games_df[c] = games_df[c].map(
                lambda x: normalize_whitespace(fix_mojibake(x))
            )

    if "genres" in games_df.columns:
        games_df["genres_list"] = games_df["genres"].map(parse_listlike)
        games_df["genres_str"] = games_df["genres_list"].map(
            lambda xs: " | ".join(xs) if xs else ""
        )

    if "release_date" in games_df.columns:
        games_df["release_date_parsed"] = games_df["release_date"].map(parse_date_any)

    for c in ["developer", "publisher"]:
        if c in games_df.columns:

            def norm_org(val):
                if pd.isna(val):
                    return ""
                lst = parse_listlike(val)
                if isinstance(lst, list) and lst:
                    return "; ".join(lst)
                return normalize_whitespace(str(val))

            games_df[c] = games_df[c].map(norm_org)

    games_df["_game_key"] = games_df["name"].map(make_game_key)

    if len(games_df):
        games_df["_rank_keep"] = (
            games_df["long_description"].notna().astype(int) * 2
            + games_df["release_date_parsed"].notna().astype(int)
        )
        games_df = (
            games_df.sort_values(
                ["_game_key", "_rank_keep", "release_date_parsed"],
                ascending=[True, False, False],
            )
            .drop_duplicates(subset=["_game_key"], keep="first")
            .drop(columns=["_rank_keep"])
            .reset_index(drop=True)
        )

    rev_keep = [
        "review",
        "hours_played",
        "helpful",
        "funny",
        "recommendation",
        "date",
        "game_name",
        "username",
    ]
    rev_keep = [c for c in rev_keep if c in reviews_df.columns]
    reviews_df = reviews_df[rev_keep].copy()

    for c in ["review", "recommendation", "game_name", "username"]:
        if c in reviews_df.columns:
            reviews_df[c] = reviews_df[c].map(
                lambda x: normalize_whitespace(fix_mojibake(x))
            )

    for c in ["hours_played", "helpful", "funny"]:
        if c in reviews_df.columns:
            reviews_df[c] = reviews_df[c].map(to_number).astype(float)

    if "date" in reviews_df.columns:
        reviews_df["review_date"] = reviews_df["date"].map(parse_date_any)

    reviews_df["_game_key"] = reviews_df["game_name"].map(make_game_key)
    reviews_df = reviews_df.drop_duplicates(
        subset=["_game_key", "username", "review"]
    ).reset_index(drop=True)

    aug_cols = [
        "_game_key",
        "genres_str",
        "developer",
        "publisher",
        "overall_player_rating",
        "short_description",
    ]
    games_aug = games_df[[c for c in aug_cols if c in games_df.columns]].copy()
    reviews_df = reviews_df.merge(games_aug, on="_game_key", how="left")

    print("Cleaned games_df:", games_df.shape)
    print("Cleaned reviews_df:", reviews_df.shape)

    return games_df, reviews_df, desc_p, rev_p
