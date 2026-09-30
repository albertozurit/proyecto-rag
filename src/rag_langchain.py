"""Primer RAG local con LangChain y Ollama."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROMPT = ChatPromptTemplate.from_template(
    """Responde a la pregunta usando unicamente el contexto proporcionado.
Si la respuesta no aparece en el contexto, indica que no hay informacion suficiente.
No inventes datos.

Contexto:
{context}

Pregunta: {question}
"""
)


def load_documents(data_dir: Path) -> list[Document]:
    """Carga documentos Markdown del directorio de conocimiento."""
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")

    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in sorted(data_dir.rglob("*.md"))
    ]
    if not documents:
        raise ValueError(f"No se encontraron documentos Markdown en {data_dir}")
    return documents


def build_retriever(data_dir: Path, embedding_model: str):
    """Divide documentos, crea embeddings y devuelve un retriever."""
    documents = load_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    top_k = int(os.getenv("RAG_TOP_K", "4"))
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def answer_question(question: str, data_dir: Path) -> tuple[str, list[Document]]:
    """Recupera contexto y genera una respuesta con el modelo local."""
    llm_model = os.getenv("OLLAMA_LLM", "qwen2.5:1.5b")
    embedding_model = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    retriever = build_retriever(data_dir, embedding_model)
    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)
    prompt = PROMPT.invoke({"context": context, "question": question})
    response = ChatOllama(model=llm_model, base_url=base_url, temperature=0).invoke(prompt)
    return str(response.content), documents


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Pregunta a un RAG local con LangChain.")
    parser.add_argument("question", help="Pregunta que se quiere responder")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directorio con documentos Markdown (por defecto: data)",
    )
    args = parser.parse_args()

    response, sources = answer_question(args.question, args.data_dir)
    print(f"\nRespuesta:\n{response}\n")
    print("Fuentes recuperadas:")
    for source in sources:
        print(f"- {source.metadata.get('source', 'desconocida')}")


if __name__ == "__main__":
    main()