import streamlit as st
from rag import answer_question


st.set_page_config(
    page_title="IT Policy Assistant",
    page_icon="📘",
    layout="centered"
)


st.title("IT Policy Assistant")

st.caption(
    "Ask a question about the company IT policy. "
    "Answers are based only on the policy document."
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# Chat input
question = st.chat_input(
    "Ask a question..."
)


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(
            question
        )


    # Generate and display assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching policy..."
        ):

            answer = answer_question(
                question
            )

        st.markdown(
            answer
        )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )