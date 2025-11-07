import sys

from gamemaster.config import DEVICE
from gamemaster.utils import print_versions_and_checksums
from gamemaster.embeddings import build_indices
from gamemaster.llm import load_llm
from gamemaster.rag import RAGEngine


def main():
    print(f"Device: {DEVICE}")
    print_versions_and_checksums()

    emb_model, use_normalize, game_index, rev_index, games_unique, reviews_df = build_indices()
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

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        engine.ask(q, show_context=False)
    else:
        engine.ask("Give me 3 co-op games with a first-person perspective", show_context=True)
        engine.ask("Name 3 games that let me drive or ride vehicles", show_context=True)
        engine.ask("Name 3 survival games that focus on resource gathering and crafting.", show_context=True)


if __name__ == "__main__":
    main()
