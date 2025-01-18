import os
import textwrap

from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

templateQA = """Using the following context, provide a clear and concise answer to the question. Focus only on information present in the context.

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=templateQA,
)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=20):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    return text_splitter.split_text(text)


def wrap_text(text, width=80):
    return textwrap.fill(text, width=width)

def save_chunks_to_file_and_store(pages, output_filename, vector_store_path):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

    all_texts = []

    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write("PDF CHUNKS\n")
        output_file.write("=" * 50 + "\n\n")

        for page_number, page in enumerate(pages, 1):
            text = page.page_content
            chunks = split_text_into_chunks(text)

            output_file.write(f"Page {page_number}:\n")
            output_file.write("-" * 30 + "\n")

            for i, chunk in enumerate(chunks, 1):
                output_file.write(f"Chunk {i}:\n")
                wrapped_chunk = wrap_text(chunk, width=80)
                output_file.write(wrapped_chunk)
                output_file.write("\n\n")
                all_texts.append(chunk)

        print(f"[Progress] Processing completed: {len(pages)} pages processed")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[Progress] Creating vector store...")
    vectorstore = FAISS.from_texts(texts=all_texts, embedding=embedding_function)
    vectorstore.save_local(vector_store_path)
    print("[Progress] Vector store creation completed")

def setup_local_llm():
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=50256,
        truncation=True,
        repetition_penalty=1.2
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm

def setup_qa_chain(vector_store_path, chain_type="stuff"):
    print("[Progress] Setting up QA chain...")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        vector_store_path,
        embedding_function,
        allow_dangerous_deserialization=True
    )

    llm = setup_local_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type=chain_type,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "document_separator": "\n",
        }
    )

    return qa_chain

def answer_question(qa_chain, question, output_file_path):
    print(f"[Progress] Processing question: {question}")

    try:
        result = qa_chain.invoke({"query": question})

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Question: {question}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Answer:\n")
            f.write(result["result"])
            f.write("\n\n" + "=" * 50 + "\n\n")

            f.write("Source Documents:\n")
            for i, doc in enumerate(result["source_documents"], 1):
                f.write(f"\nSource {i}:\n")
                f.write(wrap_text(doc.page_content))
                f.write("\n" + "-" * 30 + "\n")

        print("[Progress] Answer saved to file")
        return result

    except Exception as e:
        print(f"[Error] Failed to generate answer: {str(e)}")
        return None

def setup_chat_chain(vector_store_path):
    print("[Progress] Setting up chat chain...")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        vector_store_path,
        embedding_function,
        allow_dangerous_deserialization=True
    )

    llm = setup_local_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return chat_chain

def chat_loop(chat_chain, output_file_path):
    print("[Progress] Starting chat session. Type 'quit' to exit.")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        while True:
            question = input("\nYou: ")
            if question.lower() == 'quit':
                break

            try:
                result = chat_chain.invoke({"question": question})
                answer = result['answer']

                f.write(f"User: {question}\n")
                f.write(f"Assistant: {answer}\n")
                f.write("-" * 50 + "\n")
                f.flush()

                print(f"\nAssistant: {answer}")

            except Exception as e:
                print(f"[Error] Failed to generate response: {str(e)}")

def main(pdf_file_path, vector_store_path, query=None, chat_mode=False):
    os.makedirs("output_text_files", exist_ok=True)
    os.makedirs("output_qa", exist_ok=True)
    os.makedirs("output_chat", exist_ok=True)

    chunks_file = os.path.join("output_text_files", "pdf_chunks.txt")
    qa_file = os.path.join("output_qa", "qa_results.txt")
    chat_file = os.path.join("output_chat", "chat_history.txt")

    print("\n[Progress] Starting PDF processing...")
    pages = load_pdf(pdf_file_path)
    print(f"[Progress] PDF loaded: {len(pages)} pages found")

    save_chunks_to_file_and_store(pages, chunks_file, vector_store_path)

    if chat_mode:
        print("\n[Progress] Setting up chat system...")
        chat_chain = setup_chat_chain(vector_store_path)
        chat_loop(chat_chain, chat_file)
    elif query:
        print("\n[Progress] Setting up QA system...")
        qa_chain = setup_qa_chain(vector_store_path)

        print(f"[Progress] Processing question: {query}")
        answer_question(qa_chain, query, qa_file)

    print("\n[Progress] Process completed")
    print(f"[Output] Chunks file: {chunks_file}")
    if chat_mode:
        print(f"[Output] Chat history file: {chat_file}")
    elif query:
        print(f"[Output] QA results file: {qa_file}")


if __name__ == "__main__":
    pdf_file_path = "SCS_Lab02.pdf"
    output_filename = "output_text_files/pdf_chunks.txt"
    vector_store_path = "output_vectorstore/vector_store.index"

    print("\nSelect mode:")
    print("1. Chat mode")
    print("2. QA mode")
    mode_choice = input("Enter your choice (1 or 2): ")

    chat_mode = mode_choice == "1"
    search_query = None
    
    if not chat_mode:
        search_query = input("Enter your question: ")

    main(pdf_file_path, vector_store_path,
         query=search_query, chat_mode=chat_mode)