import sys

from .config import DEVICE
from .utils.common import print_versions_and_checksums
from .services.embeddings import load_or_build_indices
from .services.llm import load_llm
from .core.engine import RAGEngine


def main():
    print(f"Device: {DEVICE}")
    print_versions_and_checksums()

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

    import argparse
    parser = argparse.ArgumentParser(description="GameMaster CLI")
    parser.add_argument("query", nargs="*", help="Direct query to initialize with")
    parser.add_argument("--history", action="store_true", help="Enable conversation history in interactive mode")
    args = parser.parse_args()

    if args.query:
        q = " ".join(args.query)
        engine.ask(q, show_context=False)
    else:
        from .config import CONFIG
        
        print("\n" + "="*50)
        print("🎮 GameMaster CLI - Interactive Mode")
        if args.history:
            print("History: ENABLED")
        else:
            print("History: DISABLED (default)")
        print("Type 'exit' or 'quit' to stop.")
        print("Type 'reset' to clear history.")
        print("="*50 + "\n")

        history = []
        max_turns = CONFIG.get("max_history_turns", 3)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "reset":
                    history = []
                    print("[History cleared]")
                    continue

                current_history = None
                if args.history and history:
                    msg_count = max_turns * 2
                    if msg_count > 0:
                        current_history = history[-msg_count:]
                    else:
                        current_history = []
                
                answer, _, _, _ = engine.generate_rag_answer(user_input, history=current_history)
                
                print(f"GameMaster: {answer}")
                
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()

