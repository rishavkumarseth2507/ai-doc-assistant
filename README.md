# 🔎 AI Document Assistant (RAG App)

An AI-powered document question-answering system built using **LangChain, FAISS, Groq LLM, and Streamlit**.  
Upload PDFs and ask questions — the system retrieves relevant context and answers using Retrieval-Augmented Generation (RAG).

---------------------------------------------

## 🚀 Features

- 📄 Upload multiple PDF documents
- 🔍 Semantic search using FAISS vector database
- 🧠 AI-powered Q&A using Groq LLM (LLaMA models)
- 📚 Context-aware responses with page references
- 💬 Chat-like interface using Streamlit
- 🔐 Secure API key handling using `.env` / Streamlit Secrets

---------------------------------------------

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- LangChain 🦜
- FAISS (Vector DB)
- Google Generative AI Embeddings
- Groq API (LLaMA 3)
- PyPDF

---------------------------------------------

## 📂 Project Structure

ai-doc-assistant/
│
├── app.py             
├── requirements.txt   
├── .gitignore  
├── .env.example      
└──  README.md     



---------------------------------------------

## ⚙️ Setup Instructions

### 1️⃣ Clone repository
```bash
git clone https://github.com/your-username/ai-doc-assistant.git
cd ai-doc-assistant
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Add environment variables
```bash
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
```

4️⃣ Run the app
```bash
streamlit run app.py
```


---------------------------------------------
🌐 Deployment

Deploy easily using Streamlit Cloud:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect repo
4. Add secrets in dashboard
5. Deploy 🚀

---------------------------------------------
🧠 How it works
1. PDFs are loaded and split into chunks
2. Embeddings are generated using Google Generative AI
3. FAISS stores vector embeddings
4. User query is matched with relevant chunks
5. Groq LLM generates final answer using retrieved context


---------------------------------------------
📌 Example Use Cases
- Academic PDF Q&A
- Medical report analysis
- Resume screening assistant
- Research paper summarize

---------------------------------------------
👨‍💻 Author

Made with ❤️ by Rishav Kumar
M.Sc. Mathematics & Scientific Computing @ MNNIT



⭐ Support

If you like this project, give it a ⭐ on GitHub!
