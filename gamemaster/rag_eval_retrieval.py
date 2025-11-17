import json
from time import perf_counter

import pandas as pd
import torch

from .config import OUTPUT_DIR, CONFIG, DEVICE
from .embeddings import load_or_build_indices
from .llm import load_llm
from .rag import RAGEngine
from .utils import make_game_key


def load_eval_qa():
    golden_path = OUTPUT_DIR / "rag_eval_golden_set.jsonl"
    qa = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            qa.append(json.loads(line))
    return qa


def load_engine():
    emb_model, use_normalize, game_index, rev_index, games_unique, reviews_df = load_or_build_indices()
    tokenizer, model = load_llm()
    engine = RAGEngine(
        emb_model=emb_model,
        use_normalize=use_normalize,
        game_index=game_index,
        rev_index=rev_index,
        games_unique=games_unique,
        reviews_df=reviews_df,
        tokenizer=tokenizer,
        model=model,
    )
    return engine


def timed_generate_rag_answer(engine: RAGEngine, user_query: str):
    t0 = perf_counter()
    t_ret_start = perf_counter()

    stage1_hits = engine.stage1_get_games(user_query, k_probe=50)
    if not stage1_hits:
        context_text = ""
        t_ret_end = perf_counter()
        prompt, _, ctx_block = engine.build_messages(user_query, context_text)
    else:
        g_signal = engine.build_game_signal(stage1_hits)
        stage2_hits = engine.stage2_get_reviews(
            user_query, g_signal, stage1_hits, k_probe=4096
        )
        t_ret_end = perf_counter()
        context_text = engine.format_two_stage_context(stage1_hits, stage2_hits)
        prompt, _, ctx_block = engine.build_messages(user_query, context_text)

    inputs = engine.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = engine.model.generate(
            **inputs,
            max_new_tokens=CONFIG["llm_max_new_tokens"],
            temperature=CONFIG["llm_temperature"],
            do_sample=True,
            pad_token_id=engine.tokenizer.eos_token_id,
        )
    gen_only = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = engine.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    t1 = perf_counter()
    e2e_latency = t1 - t0
    retriever_latency = t_ret_end - t_ret_start
    retrieved_game_names = [hit["row"].get("name", "") for hit in stage1_hits]

    return answer, retrieved_game_names, prompt, ctx_block, e2e_latency, retriever_latency


def compute_recall_mrr(relevant_games, retrieved_game_names):
    if not relevant_games:
        return 0.0, 0.0
    rel_keys = [make_game_key(g) for g in relevant_games]
    ret_keys = [make_game_key(g) for g in retrieved_game_names]
    hit_positions = []
    for rel in rel_keys:
        for idx, rk in enumerate(ret_keys):
            if rk == rel:
                hit_positions.append(idx + 1)
                break
    recall = len(hit_positions) / len(rel_keys)
    mrr = 1.0 / min(hit_positions) if hit_positions else 0.0
    return recall, mrr


def main():
    eval_qa = load_eval_qa()
    engine = load_engine()
    results = []

    for i, item in enumerate(eval_qa, start=1):
        q = item["query"]
        print(f"[{i}/{len(eval_qa)}] Asking:", q)

        answer, retrieved_games, prompt, ctx_block, e2e_lat, ret_lat = timed_generate_rag_answer(
            engine, q
        )
        recall, mrr = compute_recall_mrr(item.get("relevant_games", []), retrieved_games)

        r = {
            "id": i,
            "query": q,
            "expected_response": item.get("expected_response", ""),
            "relevant_games": item.get("relevant_games", []),
            "model_response": answer,
            "retrieved_game_names": retrieved_games,
            "prompt": prompt,
            "context": ctx_block,
            "e2e_latency_sec": e2e_lat,
            "retriever_latency_sec": ret_lat,
            "retrieval_recall": recall,
            "retrieval_mrr": mrr,
        }
        results.append(r)

    results_jsonl_path = OUTPUT_DIR / "rag_eval_results_full.jsonl"
    with open(results_jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    results_csv_path = OUTPUT_DIR / "rag_eval_results_full.csv"
    df.to_csv(results_csv_path, index=False)

    print("Saved detailed results to:")
    print(" JSONL:", results_jsonl_path)
    print(" CSV  :", results_csv_path)

    print("\nAggregate Retrieval metrics over all questions:")
    print("  Mean Recall :", df["retrieval_recall"].mean())
    print("  Mean MRR    :", df["retrieval_mrr"].mean())

    print("\nAggregate Latencies:")
    print("  Mean end-to-end latency (s):", df["e2e_latency_sec"].mean())
    print("  Mean retriever latency (s) :", df["retriever_latency_sec"].mean())


if __name__ == "__main__":
    main()

