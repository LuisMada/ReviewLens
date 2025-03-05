import streamlit as st
import chromadb
import matplotlib.pyplot as plt
from collections import Counter
from langchain.schema import Document  # ✅ Fix: Import Document class

# Optional: Import LLM for summarization (Comment out if not using GPT)
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI  # ✅ Correct

# Hardcoded OpenAI API Key
OPENAI_API_KEY = "INSERT-OPENAI-API-KEY"

# Initialize ChromaDB
DB_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="reviews")

# Load LLM (GPT-4 or OpenAI API) for Summarization (Optional)
try:
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
except Exception:
    llm = None  # Skip if OpenAI API key is missing

def search_reviews(query, sentiment=None, top_k=10):
    """Fetch the most relevant reviews with optional sentiment filtering."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    filtered_reviews = []
    filtered_metadata = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if sentiment is None or meta["sentiment"].lower() == sentiment.lower():
            filtered_reviews.append(doc)
            filtered_metadata.append(meta)

    return filtered_reviews, filtered_metadata

def summarize_reviews(reviews):
    """Summarize a list of reviews using an LLM."""
    if not reviews or llm is None:
        return "Summarization is not available."
    
    # Convert reviews to Document objects.
    doc_objects = [Document(page_content=review) for review in reviews]
    
    chain = load_summarize_chain(llm, chain_type="stuff")
    result = chain.invoke(doc_objects)  # Invoke returns the chain’s full output.
    
    # Try to extract only the summary text from the result.
    if isinstance(result, dict):
        # Check for common keys that contain the summary.
        for key in ("output_text", "text", "output"):
            if key in result:
                return result[key]
        # Fallback: join all values if no key is found.
        return " ".join(str(value) for value in result.values())
    
    elif isinstance(result, list):
        # If the result is a list, assume each element is part of the summary.
        return " ".join(item if isinstance(item, str) else str(item) for item in result)
    
    # If it's already a string, return it directly.
    return result


def plot_sentiment_distribution(metadata):
    """Plot a pie chart showing sentiment distribution."""
    sentiments = [meta["sentiment"] for meta in metadata]
    sentiment_counts = Counter(sentiments)

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures the pie is a circle.
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="ReviewLens - AI-Powered Review Search", layout="wide")
st.title("📊 ReviewLens: AI-Powered Review Search")
st.markdown("### Search for insights from customer reviews")

query = st.text_input("🔎 Ask a question (e.g., 'Show me complaints about pricing')")

# Sentiment Filter
sentiment_filter = st.selectbox("🎭 Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
sentiment = None if sentiment_filter == "All" else sentiment_filter

if query:
    with st.spinner("🔍 Searching reviews..."):
        matches, metadata = search_reviews(query, sentiment)

        if matches:
            # 🎯 NEW: Display GPT-Generated Summary 
            if llm:
                st.subheader("📄 AI Summary of Reviews")
                summary = summarize_reviews(matches)
                st.info(summary)  # Shows summary in an info box

            # 🎯 NEW: Show sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            plot_sentiment_distribution(metadata)
            
            st.subheader("🔍 Top Matching Reviews")
            for i, review in enumerate(matches):
                st.write(f"**Review {i+1}:** {review}")
                st.write(f"📌 **Sentiment:** {metadata[i]['sentiment']}")
                st.write(f"⭐ **Score:** {metadata[i]['score']}")
                st.write(f"📅 **Timestamp:** {metadata[i]['timestamp']}")
                st.write(f"📂 **Topics:** {metadata[i]['topics']}")
                st.write("---")

        else:
            st.warning("⚠️ No matching reviews found.")
