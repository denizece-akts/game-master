import numpy as np
import pandas as pd
import torch

from ..config import CONFIG, DEVICE
from ..utils.common import clamp_sentences, clamp_chars, normalize_whitespace


class RAGEngine:
    def __init__(
        self,
        emb_model,
        use_normalize: bool,
        game_index,
        rev_index,
        games_unique: pd.DataFrame,
        reviews_df: pd.DataFrame,
        tokenizer,
        model,
    ):
        self.emb_model = emb_model
        self.use_normalize = use_normalize
        self.game_index = game_index
        self.rev_index = rev_index
        self.games_unique = games_unique
        self.reviews_df = reviews_df
        self.tokenizer = tokenizer
        self.model = model

    def _embed_one(self, text: str) -> np.ndarray:
        with torch.inference_mode():
            v = self.emb_model.encode(
                [text or ""],
                convert_to_numpy=True,
                normalize_embeddings=self.use_normalize,
                batch_size=1,
                show_progress_bar=False,
            ).astype(np.float32)
        return v

    def _embed_many(self, texts, bsz: int = 64) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), bsz):
            sub = texts[i : i + bsz]
            with torch.inference_mode():
                v = self.emb_model.encode(
                    sub,
                    convert_to_numpy=True,
                    normalize_embeddings=self.use_normalize,
                    batch_size=min(bsz, len(sub)),
                    show_progress_bar=False,
                ).astype(np.float32)
            outs.append(v)
        if outs:
            return np.vstack(outs)
        return np.zeros(
            (0, self.emb_model.get_sentence_embedding_dimension()),
            dtype=np.float32,
        )

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-12, None)

    def _faiss_search(self, index, qv: np.ndarray, k: int):
        D, I = index.search(qv.astype(np.float32), k)
        return D[0], I[0]

    def _embed_history(self, history: list) -> np.ndarray:
        if not history:
            return None
        text_blocks = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            text_blocks.append(f"{role}: {msg['content']}")
        full_text = " ".join(text_blocks)
        return self._embed_one(full_text)

    def stage1_get_games(self, user_query: str, k_probe: int = 50, history: list = None):
        w_query, w_hist = CONFIG.get("mix_games", [1.0, 0.0])
        
        q_vec = self._embed_one(user_query)
        final_vec = w_query * q_vec
        
        if history and w_hist > 0.0:
            h_vec = self._embed_history(history)
            if h_vec is not None:
                final_vec += w_hist * h_vec
        
        if self.use_normalize:
            final_vec = self._norm(final_vec)

        D, I = self._faiss_search(self.game_index, final_vec, k_probe)

        seen = set()
        hits = []
        for rank, (gid, sim) in enumerate(zip(I, D), start=1):
            if gid < 0:
                continue
            if 0 <= gid < len(self.games_unique):
                row = self.games_unique.iloc[int(gid)]
                gk = row.get("_game_key", None)
                if gk and gk in seen:
                    continue
                seen.add(gk)
                hits.append(
                    {"rank": rank, "game_id": int(gid), "score": float(sim), "row": row}
                )
                if len(hits) >= CONFIG["top_k_games"]:
                    break
        return hits

    def _make_game_text_from_row(self, g):
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
        return "\n".join(parts)

    def build_game_signal(self, selected_games):
        if not selected_games:
            return self._embed_one("")
        texts = [self._make_game_text_from_row(hit["row"]) for hit in selected_games]
        vecs = self._embed_many(texts, bsz=64)
        mean = vecs.mean(axis=0, keepdims=True)
        return self._norm(mean) if self.use_normalize else mean

    def stage2_get_reviews(
        self, user_query: str, game_signal: np.ndarray, selected_games, k_probe: int = 4096, history: list = None
    ):
        mix_conf = CONFIG.get("mix_reviews", [0.7, 0.3, 0.0])
        if len(mix_conf) == 2:
            wq, wg = mix_conf
            wh = 0.0
        else:
            wq, wg, wh = mix_conf

        q_vec = self._embed_one(user_query)
        final_vec = wq * q_vec
        
        g_vec = game_signal.astype(np.float32)
        final_vec += wg * g_vec
        
        if history and wh > 0.0:
            h_vec = self._embed_history(history)
            if h_vec is not None:
                final_vec += wh * h_vec
        
        if self.use_normalize:
            final_vec = self._norm(final_vec)

        D, I = self._faiss_search(self.rev_index, final_vec, k_probe)

        total_needed = CONFIG["top_k_reviews"]
        if total_needed <= 0:
            return []

        games_info = []
        allowed_keys = set()
        for hit in selected_games:
            row = hit["row"]
            gk = row.get("_game_key", "")
            if not gk:
                continue
            score = max(float(hit.get("score", 0.0)), 0.0)
            games_info.append({"key": gk, "score": score})
            allowed_keys.add(gk)

        if not games_info:
            return []

        candidates = []
        per_game_candidates = {g["key"]: [] for g in games_info}
        seen_rid = set()

        for rid, score in zip(I, D):
            if rid < 0:
                continue
            if 0 <= rid < len(self.reviews_df):
                if rid in seen_rid:
                    continue
                rv = self.reviews_df.iloc[int(rid)]
                gk = rv.get("_game_key", "")
                if gk not in allowed_keys:
                    continue
                seen_rid.add(rid)
                hit = {
                    "row_index": int(rid),
                    "row": rv,
                    "score": float(score),
                    "game_key": gk,
                }
                candidates.append(hit)
                per_game_candidates[gk].append(hit)

        if not candidates:
            return []

        games_with_candidates = [
            g for g in games_info if per_game_candidates.get(g["key"])
        ]
        if not games_with_candidates:
            return []

        num_games_eff = len(games_with_candidates)

        base_from_ratio = total_needed // num_games_eff
        min_per_game = max(3, base_from_ratio)
        if min_per_game * num_games_eff > total_needed:
            min_per_game = max(1, total_needed // num_games_eff)

        base_slots = min_per_game
        remaining = total_needed - base_slots * num_games_eff

        scores = [g["score"] for g in games_with_candidates]
        total_score = sum(scores)
        extra_slots = [0] * num_games_eff
        if remaining > 0:
            if total_score <= 0:
                order = list(range(num_games_eff))
            else:
                order = sorted(range(num_games_eff), key=lambda i: scores[i], reverse=True)
            idx = 0
            for _ in range(remaining):
                extra_slots[order[idx]] += 1
                idx = (idx + 1) % num_games_eff

        per_game_quota = {}
        for i, g in enumerate(games_with_candidates):
            per_game_quota[g["key"]] = base_slots + extra_slots[i]

        selected = []
        used = set()
        per_game_counts = {g["key"]: 0 for g in games_with_candidates}

        for g in games_with_candidates:
            gk = g["key"]
            quota = per_game_quota.get(gk, 0)
            cands = per_game_candidates.get(gk, [])
            for hit in cands:
                if len(selected) >= total_needed:
                    break
                rid = hit["row_index"]
                if rid in used:
                    continue
                selected.append(hit)
                used.add(rid)
                per_game_counts[gk] += 1
                if per_game_counts[gk] >= quota:
                    break
            if len(selected) >= total_needed:
                break

        if len(selected) < total_needed:
            for hit in candidates:
                if len(selected) >= total_needed:
                    break
                rid = hit["row_index"]
                if rid in used:
                    continue
                selected.append(hit)
                used.add(rid)
                gk = hit["game_key"]
                per_game_counts[gk] = per_game_counts.get(gk, 0) + 1

        return selected

    def _format_game_card(self, grow: pd.Series, rank: int, score: float) -> str:
        name = str(grow.get("name", ""))
        header = f"[GAME {rank}] Name: {name} (score={score:.4f})"
        return (
            f"{header}\n"
            f"Genres: {str(grow.get('genres_str', ''))}\n"
            f"Developer: {str(grow.get('developer', ''))}\n"
            f"Publisher: {str(grow.get('publisher', ''))}\n"
            f"Rating: {str(grow.get('overall_player_rating', ''))}\n"
            f"ShortDesc: {clamp_sentences(str(grow.get('short_description', '') or ''), 2)}\n"
            f"MinimumReq: {clamp_chars(str(grow.get('minimum_system_requirement', '') or ''), 220)}\n"
            f"RecommendedReq: {clamp_chars(str(grow.get('recommend_system_requirement', '') or ''), 220)}\n"
        )

    def _format_review_snip(self, hit, idx: int) -> str:
        rv = hit["row"]
        score = hit.get("score", None)
        game_name = str(rv.get("game_name", ""))
        header = f"[REVIEW {idx}] Game: {game_name}"
        if score is not None and not (isinstance(score, float) and np.isnan(score)):
            header += f" (score={score:.4f})"
        parts = [
            header,
            f"Review: {clamp_sentences(str(rv.get('review', '') or ''), 3)}",
            f"Recommendation: {str(rv.get('recommendation', ''))}",
        ]

        hp = rv.get("hours_played", np.nan)
        if pd.notna(hp):
            try:
                hp_val = float(hp)
                parts.append(f"Hours: {int(hp_val)}")
            except Exception:
                parts.append(f"Hours: {str(hp)}")

        hl = rv.get("helpful", np.nan)
        if pd.notna(hl):
            try:
                hl_val = float(hl)
                parts.append(f"Helpful: {int(hl_val)}")
            except Exception:
                parts.append(f"Helpful: {str(hl)}")

        return "\n".join([p for p in parts if p]) + "\n"

    def format_two_stage_context(self, stage1_games, stage2_reviews) -> str:
        lines = []
        game_keys_in_order = []
        for i, g in enumerate(stage1_games, start=1):
            grow = g["row"]
            gk = grow.get("_game_key", "")
            game_keys_in_order.append(gk)
            lines.append(self._format_game_card(grow, i, g["score"]))
        if lines:
            lines.append("")
        grouped = {gk: [] for gk in game_keys_in_order}
        for hit in stage2_reviews:
            gk = hit.get("game_key", "")
            if gk in grouped:
                grouped[gk].append(hit)
        idx = 1
        for gk in game_keys_in_order:
            hits = grouped.get(gk, [])
            hits = sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)
            for hit in hits:
                lines.append(self._format_review_snip(hit, idx))
                idx += 1
        return "\n".join(lines).strip()

    def build_messages(self, user_query: str, context_text: str, history: list = None):
        n_blocks = context_text.count("[GAME ") + context_text.count("[REVIEW ")
        ctx_block = CONFIG["context_template"].format(n=n_blocks, context=context_text)

        messages = [
            {"role": "system", "content": CONFIG["system_instruction"]},
            {"role": "system", "content": f"<CONTEXT>\n{ctx_block}\n</CONTEXT>"},
        ]
        
        if history:
            turns = len(history) // 2
            history_intro = (
                f"Conversation History: The following are the last {turns} turns. "
                "Use this history to resolve pronouns (e.g., 'it' refers to the subject of the last turn) "
                "or follow-ups in the final user query."
            )
            messages.append({"role": "system", "content": history_intro})
            
            for i in range(0, len(history), 2):
                if i+1 < len(history):
                    user_msg = history[i]
                    asst_msg = history[i+1]
                    messages.append({"role": "user", "content": user_msg['content']})
                    messages.append({"role": "assistant", "content": asst_msg['content']})
                else:
                    messages.append(history[i])

        messages.append({"role": "user", "content": user_query})

        def render(ms):
            return self.tokenizer.apply_chat_template(
                ms, tokenize=False, add_generation_prompt=True
            )

        prompt = render(messages)

        def tok_count(s: str) -> int:
            return len(self.tokenizer.encode(s, add_special_tokens=False))

        model_max = getattr(self.tokenizer, "model_max_length", 4096)
        max_new = CONFIG.get("llm_max_new_tokens", 512)
        safety = 128
        budget = model_max - max_new - safety

        if tok_count(prompt) > budget:
            ctx_lines = ctx_block.splitlines()
            while ctx_lines:
                ctx_block_trimmed = "\n".join(ctx_lines)
                messages[1]["content"] = f"<CONTEXT>\n{ctx_block_trimmed}\n</CONTEXT>"
                prompt = render(messages)
                if tok_count(prompt) <= budget:
                    ctx_block = ctx_block_trimmed
                    break
                ctx_lines.pop()

        return prompt, messages, ctx_block

    def generate_rag_answer(self, user_query: str, history: list = None):
        stage1_hits = self.stage1_get_games(user_query, k_probe=50, history=history)
        if not stage1_hits:
            context_text = ""
            prompt, _, ctx_block = self.build_messages(user_query, context_text, history=history)
        else:
            g_signal = self.build_game_signal(stage1_hits)
            stage2_hits = self.stage2_get_reviews(
                user_query, g_signal, stage1_hits, k_probe=4096, history=history
            )
            context_text = self.format_two_stage_context(stage1_hits, stage2_hits)
            prompt, _, ctx_block = self.build_messages(user_query, context_text, history)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=CONFIG["llm_max_new_tokens"],
                temperature=CONFIG["llm_temperature"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_only = output_ids[0, inputs["input_ids"].shape[1] :]
        assistant_reply = self.tokenizer.decode(
            gen_only, skip_special_tokens=True
        ).strip()
        return assistant_reply, stage1_hits, prompt, context_text

    def ask(self, question: str, show_context: bool = False, show_raw_context: bool = False, history: list = None):
        answer, _, prompt, ctx_text = self.generate_rag_answer(question, history=history)

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

        if show_context:
            print("\n" + "=" * 80)
            print("MODEL INPUT (exact prompt string passed to tokenizer)")
            print("=" * 80)
            print(prompt)

        if show_raw_context:
            print("\n" + "=" * 80)
            print("RETRIEVED CONTEXT (pre-wrapped inside <CONTEXT> in the prompt)")
            print("=" * 80)
            print(ctx_text)

