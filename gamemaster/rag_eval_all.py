import wandb

from .config import WANDB_PROJECT, WANDB_ENTITY, GOLDEN_SET_PATH, WANDB_API_KEY, WANDB_RUN_NAME
from .rag_eval_retrieval import main as retrieval_main
from .rag_eval_quality import main as quality_main


def main():
    # Login to W&B with API key
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    
    # Use custom run name if provided, otherwise auto-generate
    run_name = WANDB_RUN_NAME or f"rag_eval_pipeline_{GOLDEN_SET_PATH.stem}"
    
    # Initialize a single W&B run for the entire pipeline
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "golden_set": GOLDEN_SET_PATH.name,
            "pipeline": "retrieval_and_quality",
        },
    )
    
    try:
        print("=== Running Retrieval Evaluation ===")
        retrieval_main()
        
        print("\n=== Running Quality Evaluation ===")
        quality_main()
        
        print("\n=== Evaluation Pipeline Complete ===")
    finally:
        # Finish the W&B run
        wandb.finish()


if __name__ == "__main__":
    main()

