import json

import pandas as pd
from openai import OpenAI
import torch
import platform
import wandb
from ..config import OUTPUT_DIR, OPENAI_API_KEY, DEVICE, WANDB_PROJECT, WANDB_ENTITY

try:
    import psutil
except ImportError:
    psutil = None

MODEL_NAME = "gpt-4o-mini"


def _clamp01(x):
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0


def evaluate_all_scores(client, question: str, answer: str, context: str, expected: str):
    prompt = f"""
You are a strict evaluator.

Given:
- Question
- Model answer
- Retrieved context
- Reference (ground truth) answer

You must return a JSON object with EXACTLY these 4 keys and numeric values between 0 and 1:

{{
  "answer_relevance": <number>,
  "context_relevance": <number>,
  "groundedness": <number>,
  "ground_truth_semantic_agreement": <number>
}}

Definitions (all from 0 to 1):
- answer_relevance: how well the answer fully and directly addresses the question.
- context_relevance: how relevant the retrieved context is to answering the question.
- groundedness: how well the answer is supported ONLY by the information in the retrieved context
  (penalize hallucinated details not present in context).
- ground_truth_semantic_agreement: how semantically equivalent the model answer is to the reference answer
  (1 = same meaning, 0 = completely different).

Now evaluate:

Question:
{question}

Model answer:
{answer}

Retrieved context:
{context}

Reference (ground truth) answer:
{expected}

Respond with ONLY the JSON, no explanations, no extra text.
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict JSON-only evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return {
            "answer_relevance": _clamp01(data.get("answer_relevance", 0.0)),
            "context_relevance": _clamp01(data.get("context_relevance", 0.0)),
            "groundedness": _clamp01(data.get("groundedness", 0.0)),
            "ground_truth_semantic_agreement": _clamp01(
                data.get("ground_truth_semantic_agreement", 0.0)
            ),
        }
    except Exception as e:
        print("Error calling OpenAI or parsing JSON:", e)
        return None


def get_system_info():
    os_name = platform.system()
    os_release = platform.release()
    cpu = platform.processor() or platform.machine()
    ram_gb = None
    if psutil is not None:
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = None
    gpu_name = None
    vram_gb = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
        except Exception:
            gpu_name = None
            vram_gb = None
    return {
        "os_name": os_name,
        "os_release": os_release,
        "cpu": cpu,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "device": DEVICE,
    }


def main():

    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="rag_quality_eval",
        )
    
    results_jsonl_path = OUTPUT_DIR / "rag_eval_results_full.jsonl"
    results = []
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    client = OpenAI(api_key=OPENAI_API_KEY)

    scored_rows = []
    for r in results:
        q = r["query"]
        a = r["model_response"]
        ctx = r["context"]
        gt = r["expected_response"]

        print(f"Scoring QID {r['id']} ...")

        scores = evaluate_all_scores(client, q, a, ctx, gt)
        if scores is None:
            print("Stopping scoring due to error.")
            break

        r_scored = dict(r)
        r_scored["answer_relevance"] = scores["answer_relevance"]
        r_scored["context_relevance"] = scores["context_relevance"]
        r_scored["groundedness"] = scores["groundedness"]
        r_scored["ground_truth_semantic_agreement"] = scores[
            "ground_truth_semantic_agreement"
        ]
        scored_rows.append(r_scored)

    if not scored_rows:
        print("No rows were scored. Check API key/quota and rerun.")
        return

    df_scores = pd.DataFrame(scored_rows)
    scores_csv_path = OUTPUT_DIR / "rag_eval_results_with_quality_metrics.csv"
    df_scores.to_csv(scores_csv_path, index=False)

    mean_ans_rel = df_scores["answer_relevance"].mean()
    mean_ctx_rel = df_scores["context_relevance"].mean()
    mean_ground = df_scores["groundedness"].mean()
    mean_gt_agree = df_scores["ground_truth_semantic_agreement"].mean()

    mean_recall = df_scores["retrieval_recall"].mean()
    mean_mrr = df_scores["retrieval_mrr"].mean()

    mean_e2e = df_scores["e2e_latency_sec"].mean()
    p95_e2e = df_scores["e2e_latency_sec"].quantile(0.95)
    mean_ret = df_scores["retriever_latency_sec"].mean()
    p95_ret = df_scores["retriever_latency_sec"].quantile(0.95)

    print("\nSaved metrics to:", scores_csv_path)
    print("\nAggregate Answer Quality Metrics:")
    print("  Mean answer relevance:", mean_ans_rel)
    print("  Mean context relevance:", mean_ctx_rel)
    print("  Mean groundedness:", mean_ground)
    print("  Mean ground truth semantic agreement:", mean_gt_agree)

    print("\nRetrieval (games, Stage-1):")
    print("  Mean Recall:", mean_recall)
    print("  Mean MRR   :", mean_mrr)

    print("\nSystem Measures:")
    print("  Mean end-to-end latency (s):", mean_e2e)
    print("  P95  end-to-end latency (s):", p95_e2e)
    print("  Mean retriever latency (s):", mean_ret)
    print("  P95  retriever latency (s):", p95_ret)


    wandb.log({
        "quality/mean_answer_relevance": mean_ans_rel,
        "quality/mean_context_relevance": mean_ctx_rel,
        "quality/mean_groundedness": mean_ground,
        "quality/mean_ground_truth_agreement": mean_gt_agree,
    })

    if "type" in df_scores.columns:
        print("\nQuality Metrics by Question Type:")
        for q_type, group in df_scores.groupby("type"):
            t_ans_rel = group["answer_relevance"].mean()
            t_ground = group["groundedness"].mean()
            print(f"  {q_type.capitalize()}: AnsRel={t_ans_rel:.4f}, Ground={t_ground:.4f} (n={len(group)})")
            wandb.log({
                f"quality/{q_type}_answer_relevance": t_ans_rel,
                f"quality/{q_type}_groundedness": t_ground
            })
     

    table_data = []
    for r in scored_rows:
        table_data.append([
            r["id"],
            r["query"],
            r["expected_response"],
            r["model_response"],
            r["answer_relevance"],
            r["context_relevance"],
            r["groundedness"],
            r["ground_truth_semantic_agreement"],
        ])
    
    quality_table = wandb.Table(
        columns=["ID", "Query", "Expected Response", "Model Response", "Answer Rel", "Context Rel", "Groundedness", "GT Agreement"],
        data=table_data
    )
    wandb.log({"quality_results": quality_table})

    df_scores_no_ctx = df_scores.drop(columns=["prompt", "context"], errors="ignore")
    excel_path = OUTPUT_DIR / "rag_eval_results_no_prompt_context.xlsx"

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        sheet_name = "RAG_Eval"
        df_scores_no_ctx.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        max_width = 40
        for i, col in enumerate(df_scores_no_ctx.columns):
            col_series = df_scores_no_ctx[col].astype(str)
            max_content_len = col_series.map(len).max()
            header_len = len(str(col))
            raw_width = max(max_content_len, header_len) + 1
            width = min(raw_width, max_width)
            worksheet.set_column(i, i, width)

    print("Compact Excel saved to:", excel_path)

    sys_info = get_system_info()
    

    wandb.config.update({
        "system/os": f"{sys_info['os_name']} {sys_info['os_release']}",
        "system/cpu": sys_info["cpu"],
        "system/ram_gb": sys_info.get("ram_gb"),
        "system/device": sys_info["device"],
        "system/gpu": sys_info.get("gpu_name"),
        "system/vram_gb": sys_info.get("vram_gb"),
    })
    
    summary_txt_path = OUTPUT_DIR / "rag_eval_final_summary.txt"
    lines = []
    lines.append("System Info")
    lines.append(f"  OS: {sys_info['os_name']} {sys_info['os_release']}")
    lines.append(f"  CPU: {sys_info['cpu']}")
    if sys_info["ram_gb"] is not None:
        lines.append(f"  RAM (GB): {sys_info['ram_gb']:.2f}")
    else:
        lines.append("  RAM (GB): unknown")
    lines.append(f"  Device: {sys_info['device']}")
    if sys_info["gpu_name"] is not None:
        lines.append(f"  GPU: {sys_info['gpu_name']}")
        if sys_info["vram_gb"] is not None:
            lines.append(f"  VRAM (GB): {sys_info['vram_gb']:.2f}")
        else:
            lines.append("  VRAM (GB): unknown")
    else:
        lines.append("  GPU: none")

    lines.append("")
    lines.append("Aggregate Answer Quality Metrics:")
    lines.append(f"  Mean answer relevance: {mean_ans_rel}")
    lines.append(f"  Mean context relevance: {mean_ctx_rel}")
    lines.append(f"  Mean groundedness: {mean_ground}")
    lines.append(f"  Mean ground truth semantic agreement: {mean_gt_agree}")
    lines.append("")
    lines.append("")
    lines.append("Retrieval (games, Stage-1):")
    lines.append(f"  Mean Recall: {mean_recall}")
    lines.append(f"  Mean MRR   : {mean_mrr}")

    if "type" in df_scores.columns:
        lines.append("")
        lines.append("Breakdown by Type:")
        for q_type, group in df_scores.groupby("type"):
            t_ans_rel = group["answer_relevance"].mean()
            t_recall = group["retrieval_recall"].mean()
            lines.append(f"  {q_type.capitalize()} (n={len(group)}): AnsRel={t_ans_rel:.4f}, Recall={t_recall:.4f}")
    lines.append("")
    lines.append("System Measures:")
    lines.append(f"  Mean end-to-end latency (s): {mean_e2e}")
    lines.append(f"  P95  end-to-end latency (s): {p95_e2e}")
    lines.append(f"  Mean retriever latency (s): {mean_ret}")
    lines.append(f"  P95  retriever latency (s): {p95_ret}")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Final summary txt saved to:", summary_txt_path)


if __name__ == "__main__":
    main()

