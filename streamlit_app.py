import streamlit as st

from gamemaster.embeddings import load_or_build_indices
from gamemaster.llm import load_llm
from gamemaster.rag import RAGEngine
from gamemaster.config import DEVICE


@st.cache_resource
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


def main():
    st.set_page_config(
        page_title="GameMaster ChatBot",
        page_icon="🎮",
    )

    st.markdown("## 🎮 GameMaster ChatBot")
    st.caption(f"Device: `{DEVICE}` — Ask questions about the games.")

    engine = load_engine()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about games, co-op, survival, crafting, vehicles, etc...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        answer, stage1_hits, prompt, ctx_text = engine.generate_rag_answer(user_input)

        print("\n" + "=" * 80)
        print("QUESTION (from Streamlit)")
        print("=" * 80)
        print(user_input)

        print("\n" + "=" * 80)
        print("MODEL INPUT (exact prompt string passed to tokenizer)")
        print("=" * 80)
        print(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

