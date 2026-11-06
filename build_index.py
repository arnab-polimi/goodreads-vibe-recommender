# Copyright (c) 2026 Arnab Mondal. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying or distribution is prohibited.

import os
import re
import html
import ast
import pickle
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

KAGGLE_ZIP_PATH = os.path.join(DATA_DIR, "kaggle_10m", "goodreads_10m.zip")
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/bahramjannesarr/goodreads-book-datasets-10m"

CSV_PATH = os.path.join(DATA_DIR, "goodreads_books.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "book_embeddings.npy")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Genre inference rules for Kaggle 10M rows that don't have explicit Goodreads shelf tags
GENRE_RULES = {
    "Gothic": r"\b(gothic|macabre|victorian|brooding|manor|haunted house|dracula|frankenstein)\b",
    "Horror": r"\b(horror|terrifying|nightmare|supernatural|ghost|haunting|monster|undead|vampire|zombie|creepy|chilling)\b",
    "Mystery": r"\b(mystery|detective|murder|investigation|whodunit|crime|clues|suspect|sleuth|disappearance)\b",
    "Thriller": r"\b(thriller|suspense|conspiracy|assassin|espionage|kidnapped|stalker|deadly|hostage|psychopath)\b",
    "Science Fiction": r"\b(science fiction|sci-fi|spaceship|galaxy|alien|planet|interstellar|cyborg|android|time travel|martian|spacecraft)\b",
    "Cyberpunk": r"\b(cyberpunk|cyberspace|hacker|neon|virtual reality|megacorp|dystopian future|artificial intelligence)\b",
    "Dystopia": r"\b(dystopia|dystopian|totalitarian|post-apocalyptic|apocalypse|wasteland|regime|rebellion)\b",
    "Fantasy": r"\b(fantasy|magic|magical|wizard|dragon|sorcerer|kingdom|sword|prophecy|fae|faerie|elf|elves|spell)\b",
    "Paranormal": r"\b(paranormal|werewolf|shifter|vampires|witches|warlock|demon|immortal|fated mates)\b",
    "Romance": r"\b(romance|romantic|love story|passionate|heartbreak|lover|marriage|wedding|seduction|billionaire)\b",
    "Historical Fiction": r"\b(historical|century|world war|victorian|regency|medieval|tudor|ancient|civil war|empire)\b",
    "Philosophy": r"\b(philosophy|philosophical|existential|metaphysical|ethics|consciousness|stoic|enlightenment)\b",
    "Psychology": r"\b(psychology|psychological|psychiatrist|asylum|trauma|madness|mind|obsessive|delusion)\b",
    "Classics": r"\b(classic|masterpiece|19th century|20th century|literary classic|pulitzer|nobel)\b",
    "Nonfiction": r"\b(biography|memoir|autobiography|history of|essays|journalism|true story|scientific)\b",
}


def clean_html_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(title: str) -> str:
    t = str(title).lower()
    t = re.sub(r"\s*\(.*?\)\s*", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


def parse_genres_list(val):
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in str(val).split(",") if x.strip()]


def infer_genres_from_text(text: str) -> list[str]:
    matched = []
    text_lower = text.lower()
    for genre, pattern in GENRE_RULES.items():
        if re.search(pattern, text_lower):
            matched.append(genre)
    if not matched:
        matched.append("Fiction")
    return matched


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def ensure_kaggle_10m_downloaded():
    if os.path.exists(KAGGLE_ZIP_PATH) and os.path.getsize(KAGGLE_ZIP_PATH) > 400_000_000:
        print(f"Found local Kaggle 10M archive: {KAGGLE_ZIP_PATH}")
        return
    os.makedirs(os.path.dirname(KAGGLE_ZIP_PATH), exist_ok=True)
    print(f"Downloading bahramjannesarr/goodreads-book-datasets-10m from Kaggle...")
    with requests.get(KAGGLE_URL, stream=True) as r:
        r.raise_for_status()
        with open(KAGGLE_ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 4):
                if chunk:
                    f.write(chunk)
    print("Download complete!")


def build_data_and_index():
    ensure_kaggle_10m_downloaded()

    print("1. Loading Goodreads shelf tags lookup to enrich Kaggle 10M books...")
    genre_lookup = {}
    try:
        hf_csv = hf_hub_download("Eitanli/goodreads", "goodreads_data.csv", repo_type="dataset")
        hf_df = pd.read_csv(hf_csv)
        for _, row in hf_df.iterrows():
            k = normalize_key(row.get("Book", ""))
            glist = parse_genres_list(row.get("Genres", ""))
            if k and glist:
                genre_lookup[k] = glist
    except Exception as e:
        print(f"Warning: Could not load auxiliary genre tags: {e}")

    print("2. Reading and cleaning all CSV shards from bahramjannesarr/goodreads-book-datasets-10m...")
    chunks = []
    with zipfile.ZipFile(KAGGLE_ZIP_PATH, "r") as z:
        for info in z.infolist():
            if info.filename.startswith("book") and info.filename.endswith(".csv"):
                with z.open(info.filename) as f:
                    head_df = pd.read_csv(f, nrows=1)
                    if "Description" not in head_df.columns:
                        continue
                with z.open(info.filename) as f:
                    cols_to_read = [
                        c for c in ["Id", "Name", "Authors", "ISBN", "Rating", "PublishYear", "Publisher", "RatingDistTotal", "CountsOfReview", "Language", "Description"]
                        if c in head_df.columns
                    ]
                    df_shard = pd.read_csv(f, usecols=cols_to_read, low_memory=False)
                    lang_mask = df_shard["Language"].fillna("eng").isin(["eng", "en-US", "en-GB", "en-CA", "en"])
                    df_shard = df_shard[lang_mask & df_shard["Description"].notna()].copy()
                    df_shard["Num_Ratings_Int"] = (
                        pd.to_numeric(
                            df_shard["RatingDistTotal"].astype(str).str.replace("total:", "", regex=False),
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(int)
                    )
                    # Keep books with >= 750 Goodreads ratings so we get ~55,000+ top-quality books from the 10M dataset
                    df_shard = df_shard[df_shard["Num_Ratings_Int"] >= 750].copy()
                    chunks.append(df_shard)
                    print(f"  Loaded {info.filename}: {len(df_shard):,} popular books with descriptions")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Combined raw shards: {len(df):,} records.")

    # Clean HTML from descriptions
    print("3. Cleaning HTML descriptions and deduplicating editions...")
    df["Description"] = df["Description"].apply(clean_html_text)
    df = df[df["Description"].str.len() >= 80].copy()

    df["Book"] = df["Name"].fillna("Unknown Title").astype(str).str.strip()
    df["Author"] = df["Authors"].fillna("Unknown Author").astype(str).str.strip()
    df["Avg_Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    df["URL"] = "https://www.goodreads.com/book/show/" + df["Id"].astype(str)

    # Deduplicate editions of the same book (keep edition with highest number of ratings)
    df["norm_key"] = df["Book"].apply(normalize_key)
    df = df.sort_values("Num_Ratings_Int", ascending=False).drop_duplicates(subset=["norm_key"], keep="first")

    # Also merge any books from the 10k shelf-tagged set that weren't in the >=750 shard
    if "hf_df" in locals():
        hf_df["Description"] = hf_df["Description"].fillna("").astype(str).apply(clean_html_text)
        hf_df = hf_df[hf_df["Description"].str.len() >= 80].copy()
        hf_df["Num_Ratings_Int"] = (
            pd.to_numeric(hf_df["Num_Ratings"].astype(str).str.replace(",", "", regex=False), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        hf_df["norm_key"] = hf_df["Book"].apply(normalize_key)
        missing_hf = hf_df[~hf_df["norm_key"].isin(set(df["norm_key"]))].copy()
        if not missing_hf.empty:
            missing_hf["Id"] = 0
            missing_hf["PublishYear"] = np.nan
            missing_hf["Publisher"] = ""
            missing_hf["CountsOfReview"] = 0
            df = pd.concat(
                [
                    df[["Id", "Book", "Author", "Description", "Avg_Rating", "Num_Ratings_Int", "PublishYear", "Publisher", "CountsOfReview", "URL", "norm_key"]],
                    missing_hf[["Id", "Book", "Author", "Description", "Avg_Rating", "Num_Ratings_Int", "PublishYear", "Publisher", "CountsOfReview", "URL", "norm_key"]],
                ],
                ignore_index=True,
            )

    df = df.sort_values("Num_Ratings_Int", ascending=False).reset_index(drop=True)

    # Enrich genres using shelf lookup + automatic vibe/genre inference on descriptions
    print("4. Tagging genres and atmospheric themes across the catalog...")
    genres_list_col = []
    for _, row in df.iterrows():
        nk = row["norm_key"]
        if nk in genre_lookup:
            glist = genre_lookup[nk]
        else:
            glist = infer_genres_from_text(row["Book"] + " " + row["Description"])
        genres_list_col.append(glist)

    df["Genres"] = [str(g) for g in genres_list_col]
    df["Genres_Clean"] = [", ".join(g) for g in genres_list_col]
    df = df.drop(columns=["norm_key"])

    vibe_documents = (
        "Book Title: " + df["Book"] + " by " + df["Author"] + ". "
        + "Atmosphere, Vibe & Genres: " + df["Genres_Clean"] + ". "
        + "Synopsis: " + df["Description"]
    )

    print(f"Final deduplicated Goodreads 10M catalog size: {len(df):,} unique books!")
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved catalog to {CSV_PATH}")

    # Build TF-IDF Vectorizer & Matrix
    print("5. Building TF-IDF lexical vibe matrix...")
    tfidf = TfidfVectorizer(
        max_features=35000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(vibe_documents)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    print(f"Saved TF-IDF matrix of shape {tfidf_matrix.shape}.")

    # Build Dense Neural Embeddings using FP16 on CUDA for fast encoding
    print(f"6. Computing neural semantic embeddings using {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (FP16 accelerated)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    texts = vibe_documents.tolist()
    batch_size = 256
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            ).to(device)
            model_output = model(**encoded)
            sentence_embeddings = mean_pooling(model_output, encoded["attention_mask"])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.cpu().numpy().astype(np.float32))
            if (i // batch_size) % 25 == 0:
                print(f"  Encoded {min(i + batch_size, len(texts)):,}/{len(texts):,} books...")

    embeddings_matrix = np.vstack(all_embeddings)
    np.save(EMBEDDINGS_PATH, embeddings_matrix)
    print(f"Saved neural embeddings of shape {embeddings_matrix.shape} to {EMBEDDINGS_PATH}")
    print("Goodreads 10M Index building complete!")


if __name__ == "__main__":
    build_data_and_index()
