# -*- coding: utf-8 -*-

import os
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from hazm import Normalizer

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    input_file_path: Path = Path("digikala_comments.csv")

    work_dir: Path = Path("output_dataset")
    cleaned_file_path: Path = Path("output_dataset") / "cleaned_reviews.csv"
    stats_file_path: Path = Path("output_dataset") / "dataset_stats.json"
    stats_md_path: Path = Path("output_dataset") / "DATASET_REPORT.md"
    train_file_path: Path = Path("output_dataset") / "train.csv"
    valid_file_path: Path = Path("output_dataset") / "validation.csv"
    test_file_path: Path = Path("output_dataset") / "test.csv"
    parquet_file_path: Path = Path("output_dataset") / "cleaned_reviews.parquet"
    checkpoint_file_path: Path = Path("output_dataset") / "processing_checkpoint.json"

    batch_size: int = 5000
    num_processes: int = max(1, cpu_count() - 1)
    pool_chunksize: int = 50

    min_words: int = 3
    max_words: int = 100000

    persian_numbers: bool = True
    remove_digits: bool = False
    remove_stopwords: bool = False
    use_lemmatizer: bool = False

    # for publish-quality filtering
    remove_short_noisy_texts: bool = True
    remove_repeated_char_spam: bool = True

    def __post_init__(self):
        self.num_processes = max(1, self.num_processes)


CONFIG = Config()
NORMALIZER = Normalizer(persian_numbers=CONFIG.persian_numbers)


# =========================================================
# REGEX
# =========================================================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)

EMOJI_PATTERN = re.compile(
    "["

    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"

    "]+",
    flags=re.UNICODE
)

CONTROL_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u2060-\u206f]")
NON_TEXT_PATTERN = re.compile(r"[^A-Za-z0-9\u0600-\u06FF\u200c\s]+")
MULTISPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{4,}")  # مثلا عاااااالی


# =========================================================
# HELPERS
# =========================================================
def get_sentiment(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    try:
        text = NORMALIZER.normalize(text)
    except Exception:
        pass

    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = CONTROL_CHARS_PATTERN.sub(" ", text)

    if CONFIG.remove_digits:
        text = re.sub(r"[0-9۰-۹]+", " ", text)

    text = NON_TEXT_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text


def is_noisy_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True

    if CONFIG.remove_short_noisy_texts:
        if len(text.split()) < CONFIG.min_words:
            return True

    if CONFIG.remove_repeated_char_spam and REPEATED_CHAR_PATTERN.search(text):
        return True

    # کامنت‌هایی که تقریباً فقط عدد/نماد بوده‌اند
    letters = re.sub(r"[^\u0600-\u06FFA-Za-z]+", "", text)
    if len(letters) == 0:
        return True

    return False


def count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def process_row(row: dict) -> Optional[dict]:
    try:
        raw_comment = row.get("comment", "")
        # raw_title = row.get("title", "")

        cleaned_comment = clean_text(raw_comment)
        # cleaned_title = clean_text(raw_title)

        if not cleaned_comment or is_noisy_text(cleaned_comment):
            return None

        word_count = count_words(cleaned_comment)
        if word_count < CONFIG.min_words or word_count > CONFIG.max_words:
            return None

        rating = row.get("rating", None)
        if pd.isna(rating):
            return None

        try:
            rating_int = int(rating)
        except Exception:
            return None

        if rating_int < 1 or rating_int > 5:
            return None

        return {
            # "product_id": row.get("product_id"),
            "comment_id": row.get("comment_id"),
            "rating": rating_int,
            # "sentiment": get_sentiment(rating_int),
            # "title": raw_title,
            # "comment": raw_comment,
            # "cleaned_title": cleaned_title,
            "cleaned_comment": cleaned_comment,
            "comment_word_count": word_count,
            # "title_word_count": count_words(cleaned_title),
            "likes": row.get("likes"),
            "dislikes": row.get("dislikes"),
            "is_buyer": row.get("is_buyer"),
            # "created_at": row.get("created_at"),
        }
    except Exception:
        return None


# =========================================================
# CHECKPOINT
# =========================================================
class CheckpointManager:
    def __init__(self, checkpoint_file_path: Path):
        self.checkpoint_file_path = checkpoint_file_path
        self.data = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file_path.exists():
            try:
                with open(self.checkpoint_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"last_processed_chunk": 0}

    def save(self, last_processed_chunk: int) -> None:
        self.data["last_processed_chunk"] = last_processed_chunk
        with open(self.checkpoint_file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def last_processed_chunk(self) -> int:
        return int(self.data.get("last_processed_chunk", 0))


# =========================================================
# FILE HELPERS
# =========================================================
def ensure_dirs():
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)


def count_total_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def append_batch_to_csv(df: pd.DataFrame, output_path: Path, write_header: bool) -> None:
    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig"
    )


def deduplicate_final_output(output_path: Path) -> None:
    if not output_path.exists():
        return

    df = pd.read_csv(output_path, encoding="utf-8-sig", on_bad_lines="skip")

    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"])
    else:
        df = df.drop_duplicates()

    df.to_csv(output_path, index=False, encoding="utf-8")


# =========================================================
# STATS & REPORT
# =========================================================
def make_stats_and_report(df: pd.DataFrame):
    stats = {}
    stats["rows"] = int(len(df))
    stats["unique_products"] = int(df["product_id"].nunique()) if "product_id" in df.columns else None
    stats["unique_comments"] = int(df["comment_id"].nunique()) if "comment_id" in df.columns else None

    if "rating" in df.columns:
        stats["rating_distribution"] = df["rating"].value_counts(dropna=False).sort_index().to_dict()

    if "sentiment" in df.columns:
        stats["sentiment_distribution"] = df["sentiment"].value_counts(dropna=False).to_dict()

    if "is_buyer" in df.columns:
        stats["buyer_distribution"] = df["is_buyer"].value_counts(dropna=False).to_dict()

    if "cleaned_comment" in df.columns:
        stats["avg_comment_words"] = float(df["comment_word_count"].mean())
        stats["median_comment_words"] = float(df["comment_word_count"].median())
        stats["min_comment_words"] = int(df["comment_word_count"].min())
        stats["max_comment_words"] = int(df["comment_word_count"].max())

    if "comment_id" in df.columns:
        stats["duplicate_comment_ids"] = int(df["comment_id"].duplicated().sum())

    return stats


def write_report(stats: dict, report_path: Path):
    lines = []
    lines.append("# Persian Product Reviews Dataset")
    lines.append("")
    lines.append("A cleaned Persian product review dataset prepared for research and public release.")
    lines.append("")
    lines.append("## Statistics")
    lines.append("")

    if stats.get("rows") is not None:
        lines.append(f"- Total rows: **{stats['rows']}**")
    if stats.get("unique_products") is not None:
        lines.append(f"- Unique products: **{stats['unique_products']}**")
    if stats.get("unique_comments") is not None:
        lines.append(f"- Unique comments: **{stats['unique_comments']}**")
    if stats.get("avg_comment_words") is not None:
        lines.append(f"- Average comment length: **{stats['avg_comment_words']:.2f}** words")
        lines.append(f"- Median comment length: **{stats['median_comment_words']:.2f}** words")
        lines.append(f"- Min comment length: **{stats['min_comment_words']}** words")
        lines.append(f"- Max comment length: **{stats['max_comment_words']}** words")

    lines.append("")
    lines.append("## Labels")
    lines.append("- positive")
    lines.append("- neutral")
    lines.append("- negative")
    lines.append("")
    lines.append("## Cleaning")
    lines.append("- Persian normalization")
    lines.append("- URL and email removal")
    lines.append("- Emoji removal")
    lines.append("- Invisible/control character removal")
    lines.append("- Noise filtering")
    lines.append("- Duplicate removal")
    lines.append("- Quality filtering by length")
    lines.append("")
    lines.append("## Intended Use")
    lines.append("Sentiment analysis, text classification, and Persian NLP research.")
    lines.append("")
    lines.append("## Notes")
    lines.append("This dataset is cleaned for research use and does not include raw usernames or personal identifiers.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def split_dataset(df: pd.DataFrame, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    parts = []

    if "sentiment" in df.columns:
        grouped = df.groupby("sentiment", group_keys=False)
        train_parts = []
        valid_parts = []
        test_parts = []

        for _, g in grouped:
            g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(g)
            n_train = int(n * train_ratio)
            n_valid = int(n * valid_ratio)

            train_parts.append(g.iloc[:n_train])
            valid_parts.append(g.iloc[n_train:n_train + n_valid])
            test_parts.append(g.iloc[n_train + n_valid:])

        train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        valid_df = pd.concat(valid_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_df = df.iloc[:n_train]
        valid_df = df.iloc[n_train:n_train + n_valid]
        test_df = df.iloc[n_train + n_valid:]

    return train_df, valid_df, test_df


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    print("=" * 72)
    print("Persian Dataset Cleaner")
    print("=" * 72)
    print(f"Input file      : {CONFIG.input_file_path}")
    print(f"Output dir      : {CONFIG.work_dir}")
    print(f"Cleaned file    : {CONFIG.cleaned_file_path}")
    print(f"Batch size      : {CONFIG.batch_size}")
    print(f"Processes       : {CONFIG.num_processes}")
    print("=" * 72)

    if not CONFIG.input_file_path.exists():
        print(f"ERROR: input file not found: {CONFIG.input_file_path}")
        return

    checkpoint = CheckpointManager(CONFIG.checkpoint_file_path)
    last_chunk = checkpoint.last_processed_chunk()

    if last_chunk == 0 and CONFIG.cleaned_file_path.exists():
        print("Fresh run detected. Removing old cleaned output to avoid duplicate appends.")
        CONFIG.cleaned_file_path.unlink()

    if last_chunk > 0 and not CONFIG.cleaned_file_path.exists():
        print("WARNING: checkpoint exists but cleaned output missing. Starting fresh.")
        CONFIG.checkpoint_file_path.unlink(missing_ok=True)
        last_chunk = 0
        checkpoint = CheckpointManager(CONFIG.checkpoint_file_path)

    total_lines = count_total_lines(CONFIG.input_file_path)
    total_rows = max(0, total_lines - 1)
    total_chunks = (total_rows + CONFIG.batch_size - 1) // CONFIG.batch_size

    print(f"Total rows      : {total_rows}")
    print(f"Total chunks    : {total_chunks}")
    print("=" * 72)

    rows_written = 0
    current_chunk = 0
    header_written = CONFIG.cleaned_file_path.exists()

    chunk_iter = pd.read_csv(
        CONFIG.input_file_path,
        chunksize=CONFIG.batch_size,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    with Pool(processes=CONFIG.num_processes, maxtasksperchild=5000) as pool:
        pbar = tqdm(total=total_rows, desc="Processing rows", unit="row")

        try:
            for batch_df in chunk_iter:
                current_chunk += 1

                if current_chunk <= last_chunk:
                    pbar.update(len(batch_df))
                    continue

                if batch_df.empty:
                    checkpoint.save(current_chunk)
                    pbar.update(0)
                    continue

                batch_records = batch_df.to_dict("records")
                cleaned_rows = []

                for item in pool.imap_unordered(
                    process_row,
                    batch_records,
                    chunksize=CONFIG.pool_chunksize
                ):
                    pbar.update(1)
                    if item is not None:
                        cleaned_rows.append(item)

                if cleaned_rows:
                    processed_df = pd.DataFrame(cleaned_rows)
                    append_batch_to_csv(
                        processed_df,
                        CONFIG.cleaned_file_path,
                        write_header=not header_written
                    )
                    header_written = True
                    rows_written += len(processed_df)

                checkpoint.save(current_chunk)
                pbar.set_postfix({
                    "chunk": current_chunk,
                    "saved_rows": rows_written
                })

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            checkpoint.save(current_chunk)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            checkpoint.save(current_chunk)
            raise
        finally:
            pbar.close()

    if CONFIG.cleaned_file_path.exists():
        print("Deduplicating final output...")
        deduplicate_final_output(CONFIG.cleaned_file_path)

    if not CONFIG.cleaned_file_path.exists():
        print("No cleaned output produced. Exiting.")
        return

    print("Loading cleaned output for stats and split...")
    cleaned_df = pd.read_csv(CONFIG.cleaned_file_path, encoding="utf-8-sig", on_bad_lines="skip")

    # final quality cleanup
    cleaned_df = cleaned_df.dropna(subset=["cleaned_comment"])
    cleaned_df = cleaned_df[cleaned_df["cleaned_comment"].astype(str).str.strip() != ""]
    cleaned_df = cleaned_df.drop_duplicates(subset=["comment_id"]) if "comment_id" in cleaned_df.columns else cleaned_df.drop_duplicates()

    cleaned_df.to_csv(CONFIG.cleaned_file_path, index=False, encoding="utf-8-sig")
    try:
        cleaned_df.to_parquet(CONFIG.parquet_file_path, index=False)
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    stats = make_stats_and_report(cleaned_df)

    with open(CONFIG.stats_file_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    write_report(stats, CONFIG.stats_md_path)

    train_df, valid_df, test_df = split_dataset(cleaned_df, seed=42)
    train_df.to_csv(CONFIG.train_file_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(CONFIG.valid_file_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(CONFIG.test_file_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("DONE")
    print(f"Cleaned file    : {CONFIG.cleaned_file_path}")
    print(f"Parquet file    : {CONFIG.parquet_file_path}")
    print(f"Train split     : {CONFIG.train_file_path}")
    print(f"Validation split: {CONFIG.valid_file_path}")
    print(f"Test split      : {CONFIG.test_file_path}")
    print(f"Stats JSON      : {CONFIG.stats_file_path}")
    print(f"Report MD       : {CONFIG.stats_md_path}")
    print(f"Checkpoint file : {CONFIG.checkpoint_file_path}")
    print(f"Final rows      : {len(cleaned_df)}")
    print("=" * 72)


if __name__ == "__main__":
    freeze_support()
    main()