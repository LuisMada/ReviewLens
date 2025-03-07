# Review Analysis Dashboard

A Streamlit application for analyzing customer reviews using semantic search, category filtering, and AI-generated insights.

## Features

- **Semantic Search**: Leverages embeddings to find reviews similar to your query
- **Category Filtering**: Automatically detects relevant categories in your query and filters results
- **AI-Generated Insights**: Uses OpenAI's GPT-4o to analyze reviews and provide actionable insights
- **Interactive Visualizations**: Displays sentiment distribution and topic breakdowns
- **Hybrid Search Options**: Combine semantic search with category filtering for optimal results
- **Expandable Insights**: View complete analyses without truncation
- **Download Capability**: Save insights as text files for sharing or reference

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- ChromaDB
- LangChain
- OpenAI API key
- Sentence Transformers

### Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install streamlit pandas plotly chromadb langchain langchain-community openai sentence-transformers
   ```
3. Set up your data folder and add review CSV files (see Data Format section)

### Running the Application

1. First, index your review data:
   ```bash
   python index_reviews.py
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run app2.py
   ```

## Data Format

Your review CSV files should have the following columns:
- `content`: The text of the review
- `sentiment`: The sentiment label (e.g., "Positive", "Negative")
- `score`: A numerical score or rating
- `at`: Timestamp of the review
- `topics`: Comma-separated list of relevant categories

Place your CSV files in a folder named `data` in the root directory.

## Scripts

### index_reviews.py

```python
import os
import pandas as pd
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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
```

### app2.py

This is the main application file that implements the dashboard. It contains:

- Search functionality with both semantic and category-based approaches
- OpenAI integration for query categorization and insights generation
- Data visualization using Plotly
- Streamlit UI with interactive controls

## How It Works

1. **Indexing**: `index_reviews.py` reads your review data and stores it in ChromaDB with embeddings
2. **Searching**: When you enter a query, the app finds relevant reviews using:
   - Vector embeddings for semantic similarity
   - Category detection for filtering
3. **Analysis**: The app sends the most relevant reviews to OpenAI to generate insights
4. **Visualization**: Results are displayed in tables and charts for easy understanding

## Search Settings

In the sidebar, you can adjust:

- **Semantic Search**: Toggle vector-based search on/off
- **Category Filtering**: Enable/disable filtering by detected categories
- **Number of Results**: Adjust how many reviews to return

## Troubleshooting

- **No results found**: Try broadening your query or disabling category filtering
- **Missing embeddings**: Ensure you've run `index_reviews.py` before launching the app
- **API errors**: Verify your OpenAI API key is correct and has sufficient credits

## Contributing

Feel free to submit issues or pull requests to improve the functionality.