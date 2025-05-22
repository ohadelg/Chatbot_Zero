import json
import os

from elasticsearch_client import (
    elasticsearch_client,
    get_elasticsearch_chat_message_history,
)
from flask import current_app, render_template, stream_with_context
from functools import cache
from langchain_elasticsearch import (
    ElasticsearchStore,
    SparseVectorStrategy,
    DenseVectorStrategy,
)
from llm_integrations import get_llm

INDEX = os.getenv("ES_INDEX", "workplace-app-docs")
INDEX_CHAT_HISTORY = os.getenv(
    "ES_INDEX_CHAT_HISTORY", "workplace-app-docs-chat-history"
)
ELSER_MODEL = os.getenv("ELSER_MODEL", ".elser_model_2")
SESSION_ID_TAG = "[SESSION_ID]"
SOURCE_TAG = "[SOURCE]"
DONE_TAG = "[DONE]"

store = ElasticsearchStore(
    es_connection=elasticsearch_client,
    index_name=INDEX,
    # strategy=SparseVectorStrategy(model_id=ELSER_MODEL),
    strategy=DenseVectorStrategy(model_id=ELSER_MODEL, hybrid=True)
)


@cache
def get_lazy_llm():
    return get_llm(temperature=0.1)


@stream_with_context
def ask_question(question, session_id):
    llm = get_lazy_llm()

    yield f"data: {SESSION_ID_TAG} {session_id}\n\n"
    current_app.logger.debug("Chat session ID: %s", session_id)

    chat_history = get_elasticsearch_chat_message_history(
        INDEX_CHAT_HISTORY, session_id
    )

    if len(chat_history.messages) > 0:
        # create a condensed question
        condense_question_prompt = render_template(
            "condense_question_prompt.txt",
            question=question,
            chat_history=chat_history.messages,
        )
        condensed_question = llm.invoke(condense_question_prompt).content
    else:
        condensed_question = question

    current_app.logger.debug("Condensed question: %s", condensed_question)
    current_app.logger.debug("Question: %s", question)

    docs = store.as_retriever().invoke(condensed_question)
    current_app.logger.debug(f"Retrieved {len(docs)} documents |\n {docs}")
    if len(docs) > 0:
        for doc in docs:
            doc_source = {**doc.metadata, "page_content": doc.page_content}
            current_app.logger.debug(
                "Retrieved document passage from: %s", doc.metadata["name"]
            )
            yield f"data: {SOURCE_TAG} {json.dumps(doc_source)}\n\n"
        current_app.logger.debug("got more than 0 docs - creating prompt\n")
        qa_prompt = render_template(
            "rag_prompt.txt",
            question=question,
            docs=docs,
            chat_history=chat_history.messages,
            )
        current_app.logger.debug("QA prompt: %s", qa_prompt)
    else:
        current_app.logger.debug("No documents found for question: %s", question)
        yield f"data: {SOURCE_TAG} {json.dumps({})}\n\n"
        qa_prompt = render_template(
        "no_rag_prompt.txt",
        question=question,
        chat_history=chat_history.messages,
    )

    answer = ""
    for chunk in llm.stream(qa_prompt):
        content = chunk.content.replace("\n", " ")  # the stream can get messed up with newlines
        yield f"data: {content}\n\n"
        answer += chunk.content

    # answers = [doc.page_content for doc in docs]
    # print(f"Answer RAGs:\nreturned {len(docs)} documents.\n------------------------------------------------\n")
    # for answer in answers:
    #     print(f"{answer} \n -------------------------------------------------\n")
    #     ans = answer.replace("\n", " ")
    #     yield f"data: "+ans+f"  | {qa_prompt}"


    yield f"data: {DONE_TAG}\n\n"
    current_app.logger.debug("Answer: %s", answer)

    current_app.logger.debug(f"INDEX CHAT: {INDEX}")

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)
