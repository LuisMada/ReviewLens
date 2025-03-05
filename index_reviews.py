import os
import pandas as pd
import chromadb
from langchain_community.vectorstores import Chroma  # Updated Import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated Import

# Initialize ChromaDB
DB_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="reviews")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ensure 'data' folder exists
data_folder = "data"
if not os.path.exists(data_folder):
    print("⚠️ Data folder not found! Creating 'data' directory.")
    os.makedirs(data_folder)

# Load all CSV files from data folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

if not csv_files:
    print("⚠️ No CSV files found in 'data' folder. Please add your review dataset.")
    exit()

# Read and index data
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)

    # Check if required columns exist
    required_columns = {"content", "sentiment", "score", "at", "topics"}
    if not required_columns.issubset(df.columns):
        print(f"⚠️ Skipping {file}: Missing required columns {required_columns - set(df.columns)}")
        continue

    # Store in ChromaDB
    for i, row in df.iterrows():
        if isinstance(row["content"], str):  # Ensure valid review text
            collection.add(
                ids=[f"{file}_{i}"],  # Unique ID
                documents=[row["content"]],  # Review text
                metadatas=[{
                    "sentiment": row["sentiment"],
                    "score": row["score"],
                    "topics": row["topics"],
                    "timestamp": row["at"],
                    "source": file
                }]
            )

print("✅ All reviews indexed successfully!")
