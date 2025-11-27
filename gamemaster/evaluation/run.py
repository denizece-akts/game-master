import wandb

from ..config import WANDB_PROJECT, WANDB_ENTITY, QA_SET_PATH, WANDB_API_KEY, WANDB_RUN_NAME
from .retrieval import main as retrieval_main
from .quality import main as quality_main


def main():

    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    

    run_name = WANDB_RUN_NAME or f"rag_eval_pipeline_{QA_SET_PATH.stem}"
    

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "qa_set": QA_SET_PATH.name,
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

        wandb.finish()


if __name__ == "__main__":
    main()

