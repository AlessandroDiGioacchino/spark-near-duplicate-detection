
# Finding similar items: near-duplicate detection at scale

![Spark](https://img.shields.io/badge/Apache-Spark-orange.svg?style=flat&logo=apachespark)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat&logo=python)
![MapReduce](https://img.shields.io/badge/Paradigm-MapReduce-red.svg)

This project implements a scalable pipeline for **near-duplicate detection** in massive
textual datasets. Developed for the *Algorithms for Massive Datasets* course at the
University of Milan, it focuses on identifying similar reviews within the
**Yelp academic dataset** (~5GB of JSON data) using MinHashing and Locality-Sensitive
Hashing (LSH).

## 🎯 Project objective
The goal is to find pairs of documents with a high Jaccard similarity without falling into
the $ O( n^2 ) $ complexity trap. This is achieved by mapping high-dimensional textual
data into lower-dimensional signatures while preserving similarity.

## 🛠 Methodology & tech stack
### 1. Data processing with Apache Spark
- **Distributed computing** Leveraged Spark's RDD and DataFrame APIs to process gigabytes
  of data across distributed nodes.
- **NLP preprocessing** Implemented custom cleaning pipelines including tokenization,
  stop-word removal, shingling to convert raw text into sets of features.

### 2. The LSH pipeline
- **Shingling** Documents are converted into sets of k-shingles to capture local textual
  structure.
- **MinHashing** Large sets are compressed into short **signatures** using a family of
  min-wise independent hash functions. This step ensures that the probability of a hash
  collision equals the Jaccard similarity of the documents.
- **Locality-Sensitive Hashing (LSH)** Implemented the **banding technique** to partition
  signatures into bands. This allows the algorithm to focus only on “candidate pairs”
  that share at least one identical band, drastically reducing the search space.

### 3. Evaluation & trade-offs
- Analyzed the **S-curve** to tune the number of bands ($ b $) and rows ($ r $) to balance
  false positives and false negatives.
- Benchmarked the scalability of the Spark implementation against increasing data volumes.

## 📁 Repository structure
- `finding_similar_items.ipynb` The core implementation using PySpark and Python.
- `report.pdf` Detailed theoretical analysis, complexity study, and experimental results.
- `README.md` Project overview and documentation.

## 🚀 Installation & setup
1. **Prerequisites**
   - Apache Spark 3.x
   - Python 3.8+
   - PySpark, NumPy, Pandas
2. **Dataset**
   - Download the [Yelp academic dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset).
   - Place `yelp_academic_dataset_review.json` in the project directory.
3. **Execution**:
   Run the Jupyter Notebook:
   ```bash
   jupyter notebook finding_similar_items.ipynb
   ```

## 📈 Key insights
- **Scalability** The use of LSH reduced the number of comparisons from billions to a few thousands
  candidate pairs.
- **Precision vs recall** Demonstrated how adjusting the LSH parameters can shift the model from
  high-precision (finding only very similar items) to high-recall (finding most similar items).

## ✍️ Author
**Alessandro Di Gioacchino**  
[LinkedIn](https://www.linkedin.com/in/alessandrodigioacchino) | [GitHub](https://github.com/AlessandroDiGioacchino)
