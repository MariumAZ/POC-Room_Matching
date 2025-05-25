# 🏨 LLM-Enhanced Room Matching

This project matches hotel room names from a **reference** dataset with corresponding room names from a **supplier** dataset using a combination of Large Language Model (LLM) enrichment and vector similarity search.

By generating standardized descriptions for each room and comparing their embeddings, the system accurately identifies equivalent rooms across sources, even when naming conventions differ significantly.




## 🛠 Setup

### 🔧 Dependencies
Install required packages:

```bash
pip install pandas sentence-transformers faiss-cpu groq
```



## 🧠 Methodology

### 1. 🧼 Data Cleaning & LLM Enrichment
We use a powerful LLM (`llama3-70b-8192` from GroqCloud) to:
- Clean and standardize room names
- Extract descriptive information (amenities, bed count, non-smoking status, room size, etc.)
- Output both a `cleaned_name` and a human-readable `description`

The enrichment is done via a carefully tuned prompt that ensures consistency, clarity, and avoids failures (e.g., no “description generation failed” cases).

### 2. 🔢 Embedding Generation
Each room's cleaned name and description are embedded using the [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) model from Sentence-Transformers.

### 3. 📦 Vector Indexing with FAISS
- **Reference rooms** are embedded once (offline).
- **Supplier rooms** are embedded on the fly and queried against this index (online simulation).
- Cosine similarity is used to rank how close the supplier room is to the reference.

### 4. 🎯 Matching by Similarity
- Each reference room is matched with the most similar supplier room.
- If multiple supplier rooms have similar high scores, they are flagged for manual or LLM-based review (future step).
- The output includes the match, similarity score, and full descriptions.

