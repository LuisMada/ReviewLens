# **ReviewLens - AI-Powered Review Search**

## **Project Description**
ReviewLens is an AI-powered review search and analysis tool that leverages **ChromaDB**, **LangChain**, and **Streamlit** to provide users with a robust platform for querying customer reviews. It allows users to search, analyze, and summarize reviews while also visualizing sentiment distribution.

---

## **Features & Capabilities**
### **Core Functionalities**
- **Review Search**: Query reviews from a vector database (ChromaDB) using semantic search.
- **Sentiment Filtering**: Filter reviews based on sentiment (Positive, Neutral, Negative).
- **AI-Powered Summarization**: Generate a summary of relevant reviews using OpenAI's GPT-4.
- **Sentiment Distribution Visualization**: Display sentiment analysis results in a pie chart.
- **Review Metadata Display**: Show additional information, including rating scores, timestamps, and topics.
- **Data Ingestion & Indexing**: Automatically processes CSV review files and indexes them in ChromaDB.

### **Technologies Used**
- **ChromaDB**: A lightweight and efficient vector database for storing and retrieving reviews.
- **LangChain**: Utilized for document processing, embeddings, and integration with GPT-based summarization.
- **HuggingFace Embeddings**: Sentence-transformers (`all-MiniLM-L6-v2`) used for embedding review content.
- **Streamlit**: Web-based UI for interactive search, filtering, and visualization.
- **Matplotlib**: Used for plotting sentiment distribution charts.
- **Pandas**: Handling CSV file ingestion and preprocessing.
- **OpenAI GPT-4 (Optional)**: Summarizing reviews if an API key is provided.

---

## **Installation & Setup**
### **Prerequisites**
- Python 3.8+
- Required libraries:
  ```sh
  pip install streamlit chromadb langchain pandas matplotlib sentence-transformers openai
  ```

### **How to Run**
1. **Prepare Review Data**
   - Place review CSV files in the `data/` folder.
   - Ensure required columns are present: `content, sentiment, score, at, topics`.

2. **Index Reviews**
   - Run `index_reviews.py` to process and store reviews in ChromaDB.
   ```sh
   python index_reviews.py
   ```

3. **Launch the Web App**
   - Start the Streamlit app with:
   ```sh
   streamlit run app.py
   ```

4. **Interact with the UI**
   - Enter a query to search for relevant reviews.
   - Use the sentiment filter to refine results.
   - View AI-generated summaries and sentiment distribution.

---

## **File Structure**
```plaintext
/ReviewLens
│── app.py                 # Streamlit app for querying and visualizing reviews
│── index_reviews.py        # Script for indexing CSV reviews into ChromaDB
│── data/                   # Folder containing review CSV files
│── chroma_db/              # Persistent ChromaDB storage
```

---

## **Next Steps**
- **Improve Summarization**: Test alternative LLMs for summarization efficiency.
- **Enhance Filtering**: Add more advanced filtering (e.g., keywords, rating scores).
- **Deployment**: Deploy as a web service for broader accessibility.

---

🚀 **Enjoy using ReviewLens!**

