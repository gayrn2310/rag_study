import os
from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Folder to save Chroma DB
DB_DIRECTORY = "db"
os.makedirs(DB_DIRECTORY, exist_ok=True)  # Ensure the directory exists

# Directory to save PDFs
PDF_DIRECTORY = "pdf"
os.makedirs(PDF_DIRECTORY, exist_ok=True)  # Ensure the directory exists

# Initialize LLM, embedding model, and text splitter
cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Define the prompt template
raw_prompt = PromptTemplate.from_template("""
  <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided instructions say so. [/INST] </s>
  [INST] {input}
          Context: {context}
          Answer: 
  [/INST]
""")

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    try:
        json_content = request.json
        query = json_content.get("query", "")
        if not query:
            return {"answer": "Query cannot be empty"}, 400

        print(f"query: {query}")
        response = cached_llm.invoke(query)
        print(response)

        response_answer = {"answer": response}
        return response_answer

    except Exception as e:
        print(f"Error in /ai: {e}")
        return {"answer": f"Error: {str(e)}"}, 500


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    try:
        json_content = request.json
        query = json_content.get("query", "")
        if not query:
            return {"answer": "Query cannot be empty"}, 400

        print(f"query: {query}")

        # Load the existing vector store from DB
        print("Loading vector store from DB")
        vector_store = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embedding)

        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,
                "score_threshold": 0.5,
            },
        )

        # Create the document chain with the LLM and the prompt
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        # Run the chain with the query
        result = chain.invoke({"input": query})
        print(f"Result: {result}")

        # Prepare sources to return along with the answer
        sources = []

        if isinstance(result, dict):
            # Ensure 'context' is a list of documents
            for doc in result.get("context", []):
                if hasattr(doc, "metadata") and "sources" in doc.metadata:
                    sources.append(
                        {"sources": doc.metadata["sources"], "page_content": doc.page_content}
                    )

        response = {"answer": result.get("answer", ""), "sources": sources}
        return response

    except Exception as e:
        print(f"Error in /ask_pdf: {e}")
        return {"answer": f"Error: {str(e)}"}, 500


@app.route("/pdf", methods=["POST"])
def upload_pdf():
    print("Post /pdf called")
    try:
        # Check if a file is provided
        if "file" not in request.files:
            return {"answer": "No file part in the request"}, 400

        file = request.files["file"]

        # Check if the file has a name
        if file.filename == "":
            return {"answer": "No selected file"}, 400

        # Save the file to the PDF_DIRECTORY
        file_name = file.filename
        save_file = os.path.join(PDF_DIRECTORY, file_name)
        file.save(save_file)
        print(f"File saved to: {save_file}")

        # Extract and process the PDF content
        loader = PDFPlumberLoader(save_file)
        docs = loader.load()  # Load documents from the PDF
        print(f"Extracted {len(docs)} documents")

        # Split the documents into chunks
        chunks = text_splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks")

        # Create or update Chroma vector store
        vector_store = Chroma.from_documents(chunks, embedding, persist_directory=DB_DIRECTORY)
        vector_store.persist()  # Save the vector store to disk
        print(f"Vector store updated in {DB_DIRECTORY}")

        response = {"answer": f"Successfully processed and embedded {file_name}", "doc_len": len(docs), "chunks": len(chunks)}
        return response

    except Exception as e:
        print(f"Error in /pdf: {e}")
        return {"answer": f"Error: {str(e)}"}, 500


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
