import os
import pickle
import time
import streamlit as     st
from   dotenv    import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# LangChain modules for link loading, chunking, embedding, and QA
# ─────────────────────────────────────────────────────────────────────────────

from langchain.chat_models               import ChatOpenAI
from langchain.chains                    import ConversationalRetrievalChain

from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter             import RecursiveCharacterTextSplitter
from langchain.embeddings                import OpenAIEmbeddings
from langchain.vectorstores              import FAISS

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables (OpenAI key, etc.)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI setup
# ─────────────────────────────────────────────────────────────────────────────
# st.sidebar.title("🔗 LinkIntel")
st.markdown("<h1 style='text-align: center;'>🔗 <span style='color:#1E90FF; font-size: 48px;'>LinkIntel</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Intelligent URL Analysis</h3>", unsafe_allow_html=True)
st.sidebar.title("Enter URLs to Analyze")

# Collect up to 3 URLs from the sidebar (can be generalized later)
urls             = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_clicked  = st.sidebar.button("Process URLs")

# Placeholder for status updates and query input
main_placeholder = st.empty()

# Persistent FAISS index path (can be versioned later)
VECTORSTORE_PATH = "faiss_store_openai.pkl"

# Initialize LLM with moderate creativity (can be swapped later)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Process URLs and Build Vector Store
# ─────────────────────────────────────────────────────────────────────────────
if process_clicked:
    # Load content from provided URLs
    main_placeholder.text("🔄 Loading content from URLs...")
    # loader      = UnstructuredURLLoader(urls=urls)
    loader      = WebBaseLoader(urls)

    documents   = loader.load()

    if not documents or all(len(doc.page_content.strip()) < 100 for doc in documents):
        st.error("Loaded content is too sparse or empty. Try different URLs or switch loaders.")
        st.stop()

    st.write(f"Loaded {len(documents)} documents")
    # st.write(documents[0].page_content[:500])  # Preview first 500 characters


    # Split content into manageable chunks for embedding
    main_placeholder.text("✂️ Splitting text into chunks...")
    splitter    = RecursiveCharacterTextSplitter(
                                                    separators = ['\n\n', '\n', '.', ','],
                                                    chunk_size = 1000
                                                )
    chunks      = splitter.split_documents(documents)

    if not chunks:
        st.error("No chunks were generated. Please check the input URLs or content format.")
        st.stop()

    st.write(f"Chunks generated: {len(chunks)}")

    # Generate embeddings and build FAISS index
    main_placeholder.text("🧠 Generating embeddings and building FAISS index...")
    embeddings  = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist vectorstore for reuse across queries
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    time.sleep(2)
    main_placeholder.text("✅ Processing complete. You can now ask questions!")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Accept Query and Run Retrieval QA
# ─────────────────────────────────────────────────────────────────────────────
query          = main_placeholder.text_input("🔍 Ask a question about the content:")

if query and os.path.exists(VECTORSTORE_PATH):
    # Load stored vector index
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    # Create a conversational QA chain that retrieves context-aware answers from the vector store
    qa_chain    = ConversationalRetrievalChain.from_llm(
                                                          llm       = llm,
                                                          retriever = vectorstore.as_retriever()
                                                       )
    try:
        # Run a query and get answer + sources
        result      = qa_chain({"question": query, "chat_history": []})

        # Display answer
        st.header("🧠 Answer")
        # st.write(result.get("answer", "No answer found."))
        st.write(result["answer"])

        # Display sources (if any)
        sources   = result.get("sources", "")
        if sources:
            st.subheader("📚 Sources")
            for source in sources.split("\n"):
                st.write(source)
    except Exception as e:
        st.error(f"❌ Query failed: {str(e)}")

