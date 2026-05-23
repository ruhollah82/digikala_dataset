# 🛒 Persian Digikala Product Reviews Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dataset Size](https://img.shields.io/badge/Size-237K%20Reviews-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT%2FCC--BY--4.0-orange.svg)](LICENSE)

A complete NLP data pipeline for collecting, cleaning, and preparing **Persian e-commerce product reviews** from Digikala. Designed for machine learning, sentiment analysis, and Persian NLP research.

## 📑 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Dataset Details](#dataset-details)
- [Repository Structure](#repository-structure)
- [Data Processing & Cleaning](#data-processing--cleaning)
- [Intended Use Cases](#intended-use-cases)
- [Privacy & Ethical Notes](#privacy--ethical-notes)
- [License & Citation](#license--citation)
- [Contributing](#contributing)

## 📖 Overview

This repository provides an end-to-end pipeline to scrape, normalize, and split Persian product reviews from Digikala's public API. It includes:

- 🔍 **Crawler**: Automated collection of review comments across electronics and home appliance categories.
- 🧹 **Cleaner**: Robust text normalization, noise removal, deduplication, and dataset splitting.
- 📊 **Prepared Dataset**: Ready-to-use `train`, `validation`, and `test` splits in CSV & Parquet formats.
- 📈 **Analytics**: Automated dataset statistics and a comprehensive processing report.

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/persian-digikala-reviews.git
cd persian-digikala-reviews
pip install pandas requests tqdm hazm pyarrow
```

> 💡 `pyarrow` is optional but required for Parquet export. CSV outputs work without it.

### 2. Crawl Raw Data

```bash
python crawler.py
```

Fetches and appends raw reviews to `digikala_comments.csv`. The crawler targets public API endpoints for electronics and home appliance categories.

### 3. Clean & Split Dataset

```bash
python cleaner.py
```

Processes raw data, applies normalization, removes duplicates, and generates splits in `output_dataset/`. Supports checkpointing via `processing_checkpoint.json` to resume interrupted runs.

## 📊 Dataset Details

| Metric               | Value         |
| -------------------- | ------------- |
| **Total Rows**       | 237,855       |
| **Unique Comments**  | 237,855       |
| **Avg. Length**      | 25.86 words   |
| **Median Length**    | 15 words      |
| **Min / Max Length** | 3 / 200 words |

### 📈 Rating Distribution

| Rating |   Count |
| ------ | ------: |
| ⭐ 1   |  16,707 |
| ⭐ 2   |   6,880 |
| ⭐ 3   |  20,553 |
| ⭐ 4   |  50,780 |
| ⭐ 5   | 142,935 |

### 🧩 Dataset Splits

| Split      |          Rows |
| ---------- | ------------: |
| Train      | 190,284 (80%) |
| Validation |  23,785 (10%) |
| Test       |  23,786 (10%) |

### 📁 Schema

**Raw Dataset** (`digikala_comments.csv`)  
`product_id`, `comment_id`, `rating`, `title`, `comment`, `likes`, `dislikes`, `is_buyer`, `created_at`

**Cleaned Dataset** (`output_dataset/cleaned_reviews.csv`)  
`comment_id`, `rating`, `cleaned_comment`, `comment_word_count`, `likes`, `dislikes`, `is_buyer`

## 🗂️ Repository Structure

```text
.
├── crawler.py                         # Collects raw Digikala product comments
├── cleaner.py                         # Cleans, normalizes, and splits the dataset
├── digikala_comments.csv              # Raw crawled comments (generated)
└── output_dataset/
    ├── cleaned_reviews.csv            # Final cleaned dataset
    ├── cleaned_reviews.parquet        # Optimized Parquet version
    ├── train.csv                      # Training split
    ├── validation.csv                 # Validation split
    ├── test.csv                       # Test split
    ├── dataset_stats.json             # Automated statistics
    ├── DATASET_REPORT.md              # Human-readable processing report
    └── processing_checkpoint.json     # Resume checkpoint for interrupted runs
```

## 🧼 Data Processing & Cleaning

The `cleaner.py` pipeline applies the following steps to ensure high-quality NLP-ready text:

1. **Persian Normalization**: Standardizes characters using `Hazm`
2. **Noise Removal**: Strips URLs, emails, emojis, and control/invisible characters
3. **Text Filtering**: Removes overly short, noisy, or repeated-character spam
4. **Validation**: Ensures rating integrity and format consistency
5. **Deduplication**: Removes exact duplicate comments
6. **Splitting**: Generates stratified `train/val/test` sets (80/10/10)
7. **Checkpointing**: Saves progress to resume interrupted processing

## 🎯 Intended Use Cases

- 📝 Persian sentiment analysis & aspect-based sentiment classification
- 🏷️ Product review rating prediction
- 🔤 Persian text preprocessing & normalization benchmarking
- 🤖 Fine-tuning LLMs or training lightweight models for e-commerce NLP
- 📊 Academic research in low-resource & morphologically rich languages

## 🔒 Privacy & Ethical Notes

- Data is sourced from **publicly accessible** product review pages.
- **No personal identifiers** (usernames, emails, IPs) are collected or stored.
- The dataset is intended for **research and educational purposes only**.
- ⚖️ **Sentiment Mapping** (optional):
  - `1–2` → Negative
  - `3` → Neutral
  - `4–5` → Positive

## 📜 License & Citation

### License

This project is distributed under the [MIT License](LICENSE) for code and [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the dataset.

### Citation

If you use this dataset in your research or publications, please cite it as:

```bibtex
@misc{persian_digikala_reviews_2026,
  title={Persian Digikala Product Reviews Dataset},
  author={ruhollah82},
  year={2026},
  url={https://github.com/ruhollah82/digikala_dataset},
  note={Version 1.0.0}
}
```

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request. For major architectural changes, please discuss them in the issues tab first.

---

💡 _Built for the Persian NLP community. If this dataset helped your research, consider starring ⭐ the repository._
