

# 🧪 Model Documentation

## 📘 Methodology Overview

### 🧹 Data Cleaning and Preprocessing

To prepare the datasets for room matching, we applied several filtering and transformation steps to ensure that the comparison between reference and supplier rooms is meaningful and efficient:

1. **Invalid Room Name Removal**  
   We filtered out unusable room names from the reference dataset, such as short names, meaningless identifiers (e.g., "Room #39503141"), or names lacking descriptive value.

2. **Identifying Shared Properties (`lp_id`)**  
   We focused on `lp_id`s (property IDs) common to both datasets to ensure aligned comparisons.

3. **Filtering on Shared Properties**  
   We retained only rooms belonging to these shared properties.

4. **Grouping Room Information by Property**  
   For each shared property, we grouped room info:
   - Reference: `room_id`, `room_name`
   - Supplier: `supplier_room_id`, `supplier_room_name`, `supplier_name`

5. **Merging and Analyzing Room Counts**  
   We merged the grouped data and sorted properties by the number of reference rooms.

6. **Subsampling for Efficient Evaluation**  
   We selected properties with 2–29 rooms from both sources to balance complexity and cost.

7. **Removing Perfect Duplicates Across Sources**  
   Properties with identical room names across sources were excluded to test only non-trivial mappings.

8. **Selecting a Representative Example**  
   Property at row `28` was chosen for the proof of concept, as it contained both matching and mismatching room names.

---

## 🧠 Model Architecture

![architecture](images/architecture.jpg)

### 1. LLM Enrichment (GroqCloud LLaMA3-70B)

We used GroqCloud's `llama3-70b-8192` to generate structured descriptions for each room. It extracts:
- Room type
- Bed configuration
- View
- Amenities
- Floor level and room size

The model returns:
- A normalized `cleaned_name`
- A concise, grammatically correct `description`

### 2. Embedding Model (`all-MiniLM-L6-v2`)

We convert the cleaned name + description into vector embeddings using a lightweight sentence transformer. This captures **semantic similarity** rather than just string overlap.

### 3. FAISS Vector Store

- **Reference room embeddings**: precomputed offline  
- **Supplier rooms**: embedded and searched live  

We use **FAISS with cosine similarity** (via inner product on normalized vectors) for fast and accurate matching.

---

## 🔍 Model Evaluation Summary

We use cosine similarity between embeddings to find the closest match.

- **Total Matches Evaluated**: 18  
- ✅ **Exact Match Accuracy**: 77.78% (14/18)  
- 🟡 **Soft Accuracy** (score ≥ 0.9): 77.78% (14/18)  
- 📊 **Average Score**: 0.944  
- ⚠️ **Low-Confidence Matches** (< 0.9): 4

A high similarity score indicates a strong match.

<div align="center">
  <img src="images/match_ref.png" alt="room matching" width="600"/>
</div>

Ambiguous cases (e.g., multiple supplier rooms with similar scores) are flagged for manual or LLM-based review.

---

## 🔮 Next Steps

### 🧠 LLM-based Tie-breaker

In ambiguous cases, an LLM can compare the reference description against top-N supplier rooms to decide which one is the best match.

<div align="center">
  <img src="images/pass.png" alt="room matching" width="200"/>
</div>

### 🌍 Scaling to Multiple Suppliers

Currently tested with one supplier (Expedia), the next step is to expand to multi-supplier integration to simulate real-world use cases.
