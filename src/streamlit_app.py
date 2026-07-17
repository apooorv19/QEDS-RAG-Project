# SQLITE FIX FOR STREAMLIT CLOUD (MUST BE AT VERY TOP)
# ======================================================
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging

import streamlit as st

from memory import ChatManager
from ui import (
    render_academic_response,
    render_chat_history,
    render_header,
    render_immediate_response,
    render_sidebar,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_rag_service():
    """Create the RAG service once per app process."""

    from services import RAGService

    return RAGService()


def main() -> None:
    """Configure and run the QEDS-GPT Streamlit app."""

    st.set_page_config(page_title="QEDS-GPT", page_icon="📘", layout="wide")
    render_header()

    chat_manager = ChatManager()
    semester_filter = render_sidebar(chat_manager)
    render_chat_history(chat_manager)

    user_query = st.chat_input("Ask your question...")
    if not user_query:
        return

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.status("Starting QEDS-GPT...", expanded=True) as status:
        try:
            status.update(label="Loading Groq client and query router...")
            rag_service = get_rag_service()
            chat_manager.llm = rag_service
            status.update(label="QEDS-GPT is ready.", state="complete", expanded=False)
        except Exception as exc:
            logger.error("Startup error: %s", exc, exc_info=True)
            status.update(label="Startup failed", state="error", expanded=True)
            st.error("QEDS-GPT could not finish startup. Check the error below.")
            st.exception(exc)
            st.stop()

    context_messages = chat_manager.get_context_messages()
    category = rag_service.classify_query(user_query, context_messages)
    immediate_response = rag_service.get_immediate_response(category, user_query)
    if immediate_response is not None:
        render_immediate_response(immediate_response)
        chat_manager.add_turn(user_query, immediate_response)
        st.stop()

    render_academic_response(rag_service, chat_manager, user_query, semester_filter)


if __name__ == "__main__":
    main()
