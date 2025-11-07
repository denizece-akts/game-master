import numpy as np
import pandas as pd
import torch

from .config import CONFIG, DEVICE
from .utils import clamp_sentences, clamp_chars, normalize_whitespace


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

    def stage1_get_games(self, user_query: str, k_probe: int = 50):
        q_vec = self._embed_one(user_query)
        mix = self._norm(q_vec) if self.use_normalize else q_vec
        D, I = self._faiss_search(self.game_index, mix, k_probe)

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
        self, user_query: str, game_signal: np.ndarray, selected_games, k_probe: int = 4096
    ):
        wq, wg = CONFIG["mix_reviews"]
        q_vec = self._embed_one(user_query)
        g_vec = game_signal.astype(np.float32)
        mix = wq * q_vec + wg * g_vec
        mix = self._norm(mix) if self.use_normalize else mix

        D, I = self._faiss_search(self.rev_index, mix, k_probe)

        total_needed = CONFIG["top_k_reviews"]
        allowed_keys = {hit["row"]["_game_key"] for hit in selected_games}
        out, seen = [], set()
        for rid in I:
            if rid < 0:
                continue
            if 0 <= rid < len(self.reviews_df):
                rv = self.reviews_df.iloc[int(rid)]
                gk = rv.get("_game_key", "")
                if gk in allowed_keys and rid not in seen:
                    seen.add(rid)
                    out.append({"row_index": int(rid), "row": rv})
                    if len(out) >= total_needed:
                        break
        return out

    def _format_game_card(self, grow: pd.Series, rank: int, score: float) -> str:
        return (
            f"[GAME {rank}] score={score:.4f}\n"
            f"Name: {str(grow.get('name', ''))}\n"
            f"Genres: {str(grow.get('genres_str', ''))}\n"
            f"Developer: {str(grow.get('developer', ''))}\n"
            f"Publisher: {str(grow.get('publisher', ''))}\n"
            f"Rating: {str(grow.get('overall_player_rating', ''))}\n"
            f"ShortDesc: {clamp_sentences(str(grow.get('short_description', '') or ''), 2)}\n"
            f"MinimumReq: {clamp_chars(str(grow.get('minimum_system_requirement', '') or ''), 220)}\n"
            f"RecommendedReq: {clamp_chars(str(grow.get('recommend_system_requirement', '') or ''), 220)}\n"
        )

    def _format_review_snip(self, rv: pd.Series, idx: int) -> str:
        parts = [
            f"[REVIEW {idx}] Game: {str(rv.get('game_name', ''))}",
            f"Review: {clamp_sentences(str(rv.get('review', '') or ''), 3)}",
            f"Recommendation: {str(rv.get('recommendation', ''))}",
        ]
        hp = rv.get("hours_played", np.nan)
        parts.append(f"Hours: {int(hp)}" if pd.notna(hp) else "")
        hl = rv.get("helpful", np.nan)
        parts.append(f"Helpful: {int(hl)}" if pd.notna(hl) else "")
        return "\n".join([p for p in parts if p]) + "\n"

    def format_two_stage_context(self, stage1_games, stage2_reviews) -> str:
        lines = []
        for i, g in enumerate(stage1_games, start=1):
            lines.append(self._format_game_card(g["row"], i, g["score"]))
        for i, rh in enumerate(stage2_reviews, start=1):
            lines.append(self._format_review_snip(rh["row"], i))
        return "\n".join(lines).strip()

    def build_messages(self, user_query: str, context_text: str):
        n_blocks = context_text.count("[GAME ") + context_text.count("[REVIEW ")
        ctx_block = CONFIG["context_template"].format(n=n_blocks, context=context_text)

        messages = [
            {"role": "system", "content": CONFIG["system_instruction"]},
            {"role": "system", "content": f"<CONTEXT>\n{ctx_block}\n</CONTEXT>"},
            {"role": "user", "content": user_query},
        ]

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

    def generate_rag_answer(self, user_query: str):
        stage1_hits = self.stage1_get_games(user_query, k_probe=50)
        if not stage1_hits:
            context_text = ""
            prompt, _, ctx_block = self.build_messages(user_query, context_text)
        else:
            g_signal = self.build_game_signal(stage1_hits)
            stage2_hits = self.stage2_get_reviews(
                user_query, g_signal, stage1_hits, k_probe=4096
            )
            context_text = self.format_two_stage_context(stage1_hits, stage2_hits)
            prompt, _, ctx_block = self.build_messages(user_query, context_text)

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

    def ask(self, question: str, show_context: bool = False, show_raw_context: bool = False):
        answer, _, prompt, ctx_text = self.generate_rag_answer(question)

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
