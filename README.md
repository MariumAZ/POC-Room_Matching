# 🏨 Room Matching using LLMs


<div align="center">
  <img src="images/spot-diff.avif" alt="room matching" width="400"/>
</div>

This project matches hotel room names from a **reference** dataset with corresponding room names from a **supplier** dataset using a combination of Large Language Model (LLM) enrichment and vector similarity search.

By generating standardized descriptions for each room and comparing their embeddings, the system accurately identifies equivalent rooms across sources, even when naming conventions differ significantly.




## 🛠 Setup

### 🔧 Dependencies
Install required packages:

```bash
pip install pandas sentence-transformers faiss-cpu groq
```

### 🔑 GroqCloud API Setup
1. Sign up at https://groq.com/groqcloud/ and obtain your API key.

2. Add your API key as an environment variable:

    ✅ In Google Colab:

    Add a secret key under secrets.

    ```python
    from google.colab import userdata
   groq_token=userdata.get('secretName')
    ```
### Data

Intially we have two csv files : 
- Reference room names csv.
- Supplier room names csv. 


