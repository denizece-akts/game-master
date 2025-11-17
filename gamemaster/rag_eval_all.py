from .rag_eval_golden import main as golden_main
from .rag_eval_retrieval import main as retrieval_main
from .rag_eval_quality import main as quality_main


def main():
    golden_main()
    retrieval_main()
    quality_main()


if __name__ == "__main__":
    main()

