import streamlit as st
import pandas as pd
import plotly.express as px
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import openai
import os
import re

# Initialize OpenAI client
OPENAI_API_KEY = "OPENAIAPIKEY"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB and LangChain Vectorstore
DB_PATH = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="reviews",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

# Also keep the raw ChromaDB client for filtering capabilities
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="reviews")

CATEGORIES = [
    "Map/ Location", 
    "Payment", 
    "Rider Performance", 
    "Sanitary", 
    "Booking Experience",
    "Promo Code", 
    "Pricing", 
    "Customer Service", 
    "Management", 
    "App Performance", 
    "Generic"
]

def query_reviews(query, use_semantic_search=True, use_category_filter=True, top_k=20):
    """
    Query reviews using a hybrid approach:
    1. Semantic search using embeddings
    2. Optional category filtering
    """
    results = []
    categories = []
    
    # If using category filtering, get the categories first
    if use_category_filter:
        categories = categorize_query(query)

    if use_semantic_search:
        # Get semantically similar reviews
        semantic_results = vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # Extract documents and scores
        for doc, score in semantic_results:
            # Check if we need to filter by category
            if categories and use_category_filter:
                doc_topics = doc.metadata.get("topics", "")
                # Skip if this document doesn't contain any of the target categories
                if not any(category in doc_topics for category in categories):
                    continue
            
            results.append({
                "Review": doc.page_content,
                "Sentiment": doc.metadata.get("sentiment", ""),
                "Score": doc.metadata.get("score", ""),
                "Topics": doc.metadata.get("topics", ""),
                "Timestamp": doc.metadata.get("timestamp", ""),
                "Source File": doc.metadata.get("source", ""),
                "Relevance": float(score)
            })
    else:
        # Direct category-based search using raw document retrieval
        if categories:
            # Get all documents
            all_results = collection.get()
            
            for i, doc in enumerate(all_results["documents"]):
                metadata = all_results["metadatas"][i]
                doc_topics = metadata.get("topics", "")
                
                # Check if any of our target categories are in the document topics
                if any(category in doc_topics for category in categories):
                    results.append({
                        "Review": doc,
                        "Sentiment": metadata["sentiment"],
                        "Score": metadata["score"],
                        "Topics": metadata["topics"],
                        "Timestamp": metadata["timestamp"],
                        "Source File": metadata["source"],
                        "Relevance": 1.0  # Default relevance score
                    })
                    
                    # Limit results to top_k
                    if len(results) >= top_k:
                        break
    
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results, "No matching reviews found."
    
    # Sort by relevance score
    df_results = df_results.sort_values(by="Relevance", ascending=True)
    
    # Generate insights using OpenAI with the most relevant reviews
    top_reviews = df_results.head(min(10, len(df_results)))
    reviews_text = "\n".join([f"Review {i+1}: {review}" for i, review in enumerate(top_reviews["Review"].tolist())])
    
    prompt = f"""Given the following reviews related to the query: "{query}", provide a concise and insightful analysis:

Reviews:
{reviews_text}

Please analyze these reviews and highlight:
1. Common themes or issues mentioned
2. Sentiment patterns
3. Any specific insights relevant to the query
4. Potential actions or recommendations based on the feedback"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert review analyst who provides clear, concise, and actionable insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    insights = response.choices[0].message.content.strip()
    
    return df_results, insights, categories

def categorize_query(query):
    """Use OpenAI to categorize the query into relevant categories."""
    prompt = f"""Categorize the following user query into one or more of these categories: {', '.join(CATEGORIES)}.
    Return a comma-separated list of the relevant categories. Query: {query}"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that classifies queries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return [category.strip() for category in response.choices[0].message.content.strip().split(",")]

# Streamlit UI
st.title("Review Analysis Dashboard")

# Sidebar for search settings
st.sidebar.header("Search Settings")
use_semantic_search = st.sidebar.checkbox("Use Semantic Search", value=True, 
                                         help="Enable vector-based semantic search")
use_category_filter = st.sidebar.checkbox("Use Category Filtering", value=False, 
                                         help="Filter results by automatically detected categories")
top_k = st.sidebar.slider("Number of Results", min_value=5, max_value=50, value=20)

# User Query Input
query = st.text_input("Enter your query about the reviews:")
if query:
    st.subheader("Matching Reviews")
    df_results, insights, categories = query_reviews(
        query, 
        use_semantic_search=use_semantic_search, 
        use_category_filter=use_category_filter,
        top_k=top_k
    )
    
    # Show detected categories if using category filtering
    if use_category_filter and categories:
        st.info(f"Detected categories: {', '.join(categories)}")
    
    if not df_results.empty:
        # Display results table
        st.dataframe(df_results.drop(columns=["Relevance"]))
        
        # Insights section
        st.subheader("AI-Generated Insights")
        st.write(insights)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        # Pie Chart for Sentiment Distribution
        with col1:
            sentiment_counts = df_results["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig_sentiment = px.pie(
                sentiment_counts, 
                values='Count', 
                names='Sentiment', 
                title='Sentiment Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_sentiment)
        
        # Bar Chart for Topic Distribution
        with col2:
            # Process the Topics column to handle comma-separated values properly
            all_topics = []
            for topics_str in df_results["Topics"].dropna():
                if isinstance(topics_str, str):
                    # Split by comma and strip whitespace
                    topics_list = [t.strip() for t in topics_str.split(',')]
                    all_topics.extend(topics_list)
            
            if all_topics:
                topic_counts = pd.Series(all_topics).value_counts().reset_index()
                topic_counts.columns = ["Topic", "Count"]
                fig_topics = px.bar(
                    topic_counts, 
                    x='Topic', 
                    y='Count', 
                    title='Topic Distribution',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_topics.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_topics)
    else:
        st.info("No matching reviews found. Try modifying your query or search settings.")