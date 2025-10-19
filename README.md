# 🔗 LinkIntel: Intelligent Link Analysis

**LinkIntel** is a smart, user-friendly tool for analyzing and querying the content of any web link — not just news. Powered by LangChain, OpenAI, and FAISS, it transforms URLs into searchable knowledge, enabling users to ask questions and receive precise, context-aware answers.

![LinkIntel Demo](LinkIntel_app_UI.png)

LinkIntel allows users to input up to three URLs and extract meaningful content using LangChain's `WebBaseLoader`. The extracted content is automatically split into manageable chunks using a recursive text splitter, then embedded with OpenAI's embedding model and indexed using FAISS for fast similarity search. Users can interact with the system via a Streamlit interface, ask questions about the loaded content, and receive intelligent answers powered by LangChain's `ConversationalRetrievalChain`.

---

## 🚀 Features

- Input up to 3 URLs and fetch readable content using LangChain's `WebBaseLoader`
- Automatically split long documents into chunks for better retrieval
- Generate embeddings using OpenAI and index them with FAISS
- Ask questions and get intelligent, context-aware answers
- Store FAISS index locally for reuse across sessions
- Optional support for source-aware responses

---

## 🛠️ Installation & Setup

Clone the repository to your local machine using:

```bash
git clone https://github.com/your-username/linkintel.git
```

Navigate to the project directory:

```bash
cd linkintel
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Set up your OpenAI API key by creating a `.env` file in the project root and adding:

```bash
OPENAI_API_KEY=your_api_key_here
```

---

## 💡 Usage

Run the Streamlit app by executing:

```bash
streamlit run main.py
```

Once the app launches in your browser:

- Use the sidebar to input up to three URLs
- Click **Process URLs** to load, chunk, and embed the content
- Ask questions in the main panel and receive intelligent answers
- The FAISS index will be saved locally as a pickle file for future use

---

## 🌐 Example URLs

Try with any of these:

- https://en.wikipedia.org/wiki/Artificial_intelligence
- https://www.bbc.com/news/technology-67186794
- https://www.nytimes.com/news/articles/aiwk

---

## 📁 Project Structure

- `main.py`: Streamlit app entry point
- `requirements.txt`: Python dependencies
- `faiss_store_openai.pkl`: Serialized FAISS index
- `.env`: Stores your OpenAI API key

---

## 🧠 Powered By

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

---

## 📌 Notes

LinkIntel is designed to work with any readable URL, not just financial or news articles. For dynamic or JavaScript-heavy pages, consider extending the loader with Selenium or Puppeteer. Contributions and feedback are welcome!