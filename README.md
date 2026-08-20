
# Finding similar items: near-duplicate detection at scale

![Spark](https://img.shields.io/badge/Apache-Spark-orange.svg?style=flat&logo=apachespark)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)
![MapReduce](https://img.shields.io/badge/Paradigm-MapReduce-red.svg)

This project implements a scalable pipeline for **near-duplicate detection** in massive
textual datasets. Developed for the *Algorithms for Massive Datasets* course at the
University of Milan, it focuses on identifying similar reviews within the
**Yelp academic dataset** (~5GB of JSON data) using MinHashing and Locality-Sensitive
Hashing (LSH).

## 🎯 Project objective
The goal is to find pairs of documents with a high Jaccard similarity without falling into
the $O( n^2 )$ complexity trap. This is achieved by mapping high-dimensional textual
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
- Analyzed the **S-curve** to tune the number of bands ($b$) and rows ($r$) to balance
  false positives and false negatives.
- Benchmarked the scalability of the Spark implementation against increasing data volumes.

## 📁 Repository structure
- `src/spark_near_duplicates/` The reusable Python package and Spark pipeline.
- `finding_similar_items.ipynb` The original exploratory implementation.
- `report.tex` The theoretical analysis, complexity study, and results.
- `tests/` Unit and local-Spark integration tests.
- `README.md` Project overview and documentation.

## 🚀 Installation & setup
1. **Prerequisites**
   - Java supported by Apache Spark
   - Python 3.10+
2. **Install**
   ```bash
   python -m pip install -e .
   ```
   For tests and notebook development, use
   `python -m pip install -e ".[dev]"`.
3. **Dataset**
   - Download the [Yelp academic dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset).
   - Place `yelp_academic_dataset_review.json` in the project directory.
4. **Execution**
   Create a Spark DataFrame and pass it to the public API shown below. The
   original notebook remains available as exploratory course material.

## DataFrame API
The reusable pipeline keeps candidate generation and similarity scoring inside
Spark; it does not collect candidate pairs on the driver:

```python
from spark_near_duplicates import find_near_duplicates

matches = find_near_duplicates(
    reviews,
    id_column='review_id',
    text_column='text',
    shingle_size=5,
    num_hashes=128,
    num_bands=32,
    similarity_threshold=0.8,
)

matches.write.mode( 'overwrite' ).parquet( 'near-duplicate-pairs' )
```

`find_near_duplicates` is lazy: the final write (or another caller-selected
Spark action) executes the distributed DataFrame plan.

The input must contain distinct document identifiers and text. Identifier and
text columns default to `document_id` and `text`; use `id_column` and
`text_column` to select alternatives. Identifiers are cast to strings, and rows
with null identifiers, null text, or normalized text shorter than
`shingle_size` are omitted. Identifier uniqueness is a caller responsibility so
that validating it does not trigger an eager Spark job.

The returned schema is stable:

| Column                 | Spark type | Meaning                              |
| ---------------------- | ---------- | ------------------------------------ |
| `left_id`              |  `string`  | Lexicographically smaller identifier |
| `right_id`             |  `string`  | Lexicographically larger identifier  |
| `estimated_similarity` |  `double`  | MinHash similarity estimate          |
| `exact_similarity`     |  `double`  | Exact Jaccard similarity             |

`shingle_size`, `num_hashes`, and `num_bands` must be positive integers, and
`num_bands` must divide `num_hashes`. `similarity_threshold` is inclusive and
must be between 0 and 1. `seed` must be an unsigned 64-bit integer. The supported
package-level API consists only of `find_near_duplicates`; shingling, MinHash,
and LSH helpers are internal implementation details.

## 📈 Key insights
- **Scalability** The use of LSH reduced the number of comparisons from billions to a few thousands
  candidate pairs.
- **Precision vs recall** Demonstrated how adjusting the LSH parameters can shift the model from
  high-precision (finding only very similar items) to high-recall (finding most similar items).

## ✍️ Author
**Alessandro Di Gioacchino**
[LinkedIn](https://www.linkedin.com/in/alessandrodigioacchino) | [GitHub](https://github.com/AlessandroDiGioacchino)
