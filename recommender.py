# Copyright (c) 2026 Arnab Mondal. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying or distribution is prohibited.

import os
import ast
import gc
import re
import pickle
import warnings
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "goodreads_books.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "book_embeddings.npy")
COMPACT_CSV_PATH = os.path.join(DATA_DIR, "goodreads_books_compact.csv")
COMPACT_EMB_PATH = os.path.join(DATA_DIR, "book_embeddings_compact.npy")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
COLAB_ZIP_CANDIDATES = [
    os.path.join(BASE_DIR, "goodreads_index_files.zip"),
    os.path.join(DATA_DIR, "goodreads_index_files.zip"),
]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Curated vibe-to-genre & keyword expansion map so vibe words strongly align with Goodreads shelves/genres and themes
VIBE_EXPANSIONS = {
    "gothic": ["gothic", "dark", "victorian", "haunting", "brooding", "macabre", "manor", "castle", "shadows"],
    "horror": ["horror", "terrifying", "eerie", "supernatural", "nightmare", "creepy", "ghosts", "monsters"],
    "mysterious": ["mystery", "mysterious", "enigmatic", "secrets", "suspense", "whodunit", "detective", "puzzle"],
    "lonely": ["lonely", "isolation", "solitude", "melancholy", "alienation", "outcast", "quiet", "alone", "desolate"],
    "cozy": ["cozy", "warm", "comforting", "wholesome", "gentle", "heartwarming", "small town", "tea"],
    "dark academia": ["dark academia", "secret society", "university", "boarding school", "classics", "obsessive", "intellectual", "gothic"],
    "cyberpunk": ["cyberpunk", "dystopia", "science fiction", "neon", "artificial intelligence", "hackers", "futuristic"],
    "whimsical": ["whimsical", "magical", "fairy tale", "enchanting", "quirky", "wonder", "fantastical"],
    "melancholic": ["melancholy", "sad", "grief", "bittersweet", "poignant", "nostalgic", "emotional"],
    "epic": ["epic", "adventure", "high fantasy", "quest", "worldbuilding", "mythology", "sweeping"],
    "romantic": ["romance", "love", "passionate", "yearning", "slow burn", "heartbreak", "intimate"],
    "thrilling": ["thriller", "fast-paced", "suspense", "adrenaline", "twists", "gripping", "danger"],
    "philosophical": ["philosophy", "existential", "thought-provoking", "introspective", "meaning of life", "literary fiction"],
}

_NUMBER_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def _normalize_title(raw_title: str) -> str:
    t = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", raw_title.lower())
    t = t.split(":")[0].split(" - ")[0]
    words = [
        _NUMBER_WORDS.get(w, w)
        for w in re.sub(r"[^a-z0-9]+", " ", t).split()
        if w not in {"the", "a", "an"}
    ]
    return " ".join(words)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class VibeRecommender:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(COMPACT_CSV_PATH) or not os.path.exists(COMPACT_EMB_PATH):
            if not os.path.exists(CSV_PATH) or not os.path.exists(EMBEDDINGS_PATH):
                self._maybe_extract_colab_zip()
            if not os.path.exists(CSV_PATH) and not os.path.exists(COMPACT_CSV_PATH):
                from build_index import build_data_and_index
                build_data_and_index()
            if os.path.exists(CSV_PATH) and os.path.exists(EMBEDDINGS_PATH):
                self._build_compact_catalog()

        active_csv = COMPACT_CSV_PATH if os.path.exists(COMPACT_CSV_PATH) else CSV_PATH
        active_emb = COMPACT_EMB_PATH if os.path.exists(COMPACT_EMB_PATH) else EMBEDDINGS_PATH

        needed_cols = ["Book", "Author", "Description", "Avg_Rating", "Num_Ratings_Int", "URL", "Genres_Clean"]
        self.df = pd.read_csv(
            active_csv,
            usecols=lambda c: c in needed_cols,
            dtype={
                "Avg_Rating": np.float32,
                "Num_Ratings_Int": np.int32,
                "Author": "category",
                "Genres_Clean": "category",
            },
            converters={"Description": lambda s: s[:600] if isinstance(s, str) else ""},
            low_memory=False,
        )
        self.embeddings = (
            np.load(active_emb, mmap_mode="r") if os.path.exists(active_emb) else None
        )

        with open(TFIDF_PATH, "rb") as f:
            self.tfidf = pickle.load(f)

        self.has_aligned_embeddings = (
            self.embeddings is not None and self.embeddings.shape[0] == len(self.df)
        )
        self.emb_tensor = (
            torch.from_numpy(self.embeddings) if self.has_aligned_embeddings else None
        )

        if not self.has_aligned_embeddings and os.path.exists(TFIDF_MATRIX_PATH):
            with open(TFIDF_MATRIX_PATH, "rb") as f:
                self.tfidf_matrix = pickle.load(f)
        else:
            self.tfidf_matrix = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        # Precompute Bayesian-weighted rating + log10-popularity quality score (0 to 1)
        ratings = np.clip(self.df["Avg_Rating"].fillna(0.0).values.astype(np.float32), 0.0, 5.0)
        raw_counts = np.clip(self.df["Num_Ratings_Int"].fillna(0).values.astype(np.float32), 0.0, None)
        self.num_ratings_arr = raw_counts
        bayesian_rating = (raw_counts * ratings + 500.0 * 3.80) / (raw_counts + 500.0)
        pop_norm = np.clip((np.log10(raw_counts + 1.0) - 2.0) / 3.8, 0.0, 1.0)
        self.quality_score = np.nan_to_num(
            (bayesian_rating / 5.0) * 0.35 + pop_norm * 0.65,
            nan=0.0,
        ).astype(np.float32)

        # Precompute top popular books for the "Similar by Title" selector
        top_pop_df = self.df.nlargest(5000, "Num_Ratings_Int")
        self.popular_titles = top_pop_df["Book"].dropna().drop_duplicates().tolist()

        # Extract top genres from most popular books
        all_genres = {}
        sample_genres = top_pop_df["Genres_Clean"].astype(str).fillna("Fiction").str.split(", ")
        for glist in sample_genres:
            for g in glist:
                if g:
                    all_genres[g] = all_genres.get(g, 0) + 1
        self.popular_genres = [
            g for g, count in sorted(all_genres.items(), key=lambda x: x[1], reverse=True)
            if count >= 10
        ]
        gc.collect()

    @staticmethod
    def _build_compact_catalog():
        """Streams the raw 1.14M CSV in low-memory chunks, removes duplicate editions and unrated (<25 rating) noise,
        and writes a compact catalog so RAM stays under 700 MB on Streamlit Community Cloud."""
        needed_cols = ["Book", "Author", "Description", "Avg_Rating", "Num_Ratings_Int", "URL", "Genres_Clean"]
        kept_chunks = []
        kept_indices = []
        row_offset = 0

        for chunk in pd.read_csv(
            CSV_PATH,
            usecols=lambda c: c in needed_cols,
            dtype={"Avg_Rating": np.float32, "Num_Ratings_Int": np.int32},
            converters={"Description": lambda s: s[:600] if isinstance(s, str) else ""},
            chunksize=200_000,
            low_memory=False,
        ):
            n_chunk = len(chunk)
            # Keep volumes with >= 25 ratings and meaningful descriptions
            mask = (chunk["Num_Ratings_Int"].values >= 25) & (chunk["Description"].str.len().values >= 60)
            if mask.any():
                sub = chunk[mask].copy()
                sub["_orig_idx"] = np.where(mask)[0] + row_offset
                kept_chunks.append(sub)
            row_offset += n_chunk

        if not kept_chunks:
            return

        df_all = pd.concat(kept_chunks, ignore_index=True)
        del kept_chunks
        gc.collect()

        # Prefer primary editions with clean titles, rich genre tags, and full non-boilerplate descriptions
        desc_lower = df_all["Description"].str.lower()
        boiler_mask = desc_lower.str.startswith(
            ("in celebration of", "penguin classics", "this edition", "includes a new", "please visit www")
        )
        desc_len_bonus = np.where(df_all["Description"].str.len().values >= 180, 1.35, 0.85)
        has_junk_suffix = df_all["Book"].astype(str).str.contains(
            r"\((?:step-up|with reproduction|illustrated|abridged|adapted|large print|audio)",
            case=False,
            regex=True,
        ).values
        genre_count_bonus = 1.0 + 0.08 * df_all["Genres_Clean"].astype(str).str.count(",")

        df_all["_sort_score"] = (
            np.log1p(df_all["Num_Ratings_Int"].values.astype(np.float64))
            * desc_len_bonus
            * genre_count_bonus
            * np.where(boiler_mask, 0.55, 1.0)
            * np.where(has_junk_suffix, 0.40, 1.0)
        )
        df_all["_norm_key"] = [
            _normalize_title(t) for t in df_all["Book"].astype(str)
        ]
        df_all = df_all[df_all["_norm_key"] != ""].copy()
        df_all = (
            df_all.sort_values("_sort_score", ascending=False)
            .drop_duplicates(subset=["_norm_key"], keep="first")
            .sort_values("Num_Ratings_Int", ascending=False)
            .reset_index(drop=True)
        )

        orig_indices = df_all["_orig_idx"].values
        df_all = df_all.drop(columns=["_orig_idx", "_sort_score", "_norm_key"])
        df_all.to_csv(COMPACT_CSV_PATH, index=False)

        raw_emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        compact_emb = np.empty((len(orig_indices), raw_emb.shape[1]), dtype=np.float16)
        step = 25_000
        for s in range(0, len(orig_indices), step):
            e = min(s + step, len(orig_indices))
            compact_emb[s:e] = raw_emb[orig_indices[s:e]]
        np.save(COMPACT_EMB_PATH, compact_emb)
        del raw_emb, compact_emb, df_all
        gc.collect()

        # On Linux containers (Streamlit Cloud), remove the raw 1.8 GB uncompacted files to free disk space
        if os.name != "nt":
            for raw_path in (CSV_PATH, EMBEDDINGS_PATH, TFIDF_MATRIX_PATH):
                if os.path.exists(raw_path):
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass

    @staticmethod
    def _maybe_extract_colab_zip():
        chunks_dir = os.path.join(DATA_DIR, "chunks")
        assembled_zip = os.path.join(DATA_DIR, "goodreads_index_files.zip")
        if os.path.isdir(chunks_dir) and not os.path.exists(assembled_zip):
            parts = sorted(
                os.path.join(chunks_dir, p)
                for p in os.listdir(chunks_dir)
                if p.startswith("index.zip.part")
            )
            if parts:
                with open(assembled_zip, "wb") as out_f:
                    for part in parts:
                        with open(part, "rb") as in_f:
                            while True:
                                buf = in_f.read(8 * 1024 * 1024)
                                if not buf:
                                    break
                                out_f.write(buf)
        for zip_path in COLAB_ZIP_CANDIDATES:
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        names = zf.namelist()
                        has_emb = "book_embeddings.npy" in names
                        for member in names:
                            if has_emb and member == "tfidf_matrix.pkl":
                                continue
                            target_path = os.path.join(DATA_DIR, member)
                            if not os.path.exists(target_path):
                                zf.extract(member, DATA_DIR)
                    print(f"Extracted index files from {zip_path} into {DATA_DIR}")
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
                    break
                except Exception as e:
                    print(f"Warning: Could not extract {zip_path}: {e}")

    @staticmethod
    def _safe_parse_list(val):
        if pd.isna(val):
            return []
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in str(val).split(",") if x.strip()]

    def expand_vibe_query(self, vibe_query: str) -> str:
        """Enriches user vibe input with atmospheric synonyms so both neural & lexical search hit strong matches."""
        q_lower = vibe_query.lower()
        expanded_terms = []
        for key, synonyms in VIBE_EXPANSIONS.items():
            if key in q_lower:
                expanded_terms.extend(synonyms[:5])
        if expanded_terms:
            unique_terms = ", ".join(dict.fromkeys(expanded_terms))
            return f"Atmosphere, Vibe & Genres: {vibe_query}. Themes: {unique_terms}."
        return f"Atmosphere, Vibe & Genres: {vibe_query}."

    def encode_query(self, text: str) -> np.ndarray:
        with torch.no_grad():
            encoded = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded)
            emb = mean_pooling(model_output, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().astype(np.float32)[0]

    def extract_matching_highlights(self, vibe_query: str, row: pd.Series) -> list[str]:
        """Finds keywords/themes from the user's vibe query and expanded terms that appear in the book's genres or description."""
        raw_tokens = [
            w.strip().lower()
            for w in re.split(r"[,.;\s]+", vibe_query)
            if len(w.strip()) >= 3 and w.strip().lower() not in {"and", "the", "with", "for", "but", "like", "very", "book", "vibe"}
        ]
        candidate_terms = list(raw_tokens)
        for key, synonyms in VIBE_EXPANSIONS.items():
            if key in vibe_query.lower():
                candidate_terms.extend(synonyms)

        doc_text = (str(row["Genres_Clean"]) + " " + str(row["Description"])).lower()
        matched = []
        seen = set()
        for term in candidate_terms:
            if term not in seen and re.search(r"\b" + re.escape(term) + r"\w*\b", doc_text):
                matched.append(term)
                seen.add(term)
            if len(matched) >= 8:
                break
        return matched

    def _compute_semantic_scores_chunked(self, q_emb: np.ndarray) -> np.ndarray:
        """Computes cosine similarity in chunks over the read-only mmap to keep peak RAM negligible."""
        n = len(self.df)
        scores = np.empty(n, dtype=np.float32)
        chunk_size = 100_000
        with torch.no_grad():
            q_t = torch.from_numpy(q_emb).to(self.emb_tensor.dtype)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_out = torch.mv(self.emb_tensor[start:end], q_t).float().numpy()
                np.clip(np.nan_to_num(chunk_out, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0, out=scores[start:end])
        return scores

    def _genre_mask(self, genres_list: list[str]) -> np.ndarray:
        pat = "|".join(re.escape(g) for g in genres_list)
        col = self.df["Genres_Clean"]
        if hasattr(col, "cat"):
            cat_matches = np.asarray(col.cat.categories.astype(str).str.contains(pat, regex=True), dtype=bool)
            codes = col.cat.codes.values
            valid = codes >= 0
            out = np.zeros(len(self.df), dtype=bool)
            out[valid] = cat_matches[codes[valid]]
            return out
        return col.fillna("").astype(str).str.contains(pat, regex=True).values

    def _fast_category_genre_boost(self, vibe_query: str, expanded_query: str) -> np.ndarray:
        """Computes genre overlap across all books in <1ms using categorical dictionary lookup."""
        col = self.df["Genres_Clean"]
        q_Tokens = [
            w.lower() for w in re.split(r"[,.;\s]+", vibe_query + " " + expanded_query)
            if len(w) >= 4 and w.lower() not in {"atmosphere", "vibe", "genres", "themes", "with", "that", "from", "book"}
        ]
        unique_tokens = list(dict.fromkeys(q_Tokens))[:12]
        if not unique_tokens or not hasattr(col, "cat"):
            return np.zeros(len(self.df), dtype=np.float32)
        cats_lower = col.cat.categories.astype(str).str.lower()
        cat_boost = np.zeros(len(cats_lower), dtype=np.float32)
        for w in unique_tokens:
            stem = w[:5] if len(w) >= 6 else w
            cat_boost += np.asarray(cats_lower.str.contains(stem, regex=False), dtype=np.float32) * 0.06
        cat_boost = np.clip(cat_boost, 0.0, 0.22)
        codes = col.cat.codes.values
        return np.where(codes >= 0, cat_boost[codes], 0.0).astype(np.float32)

    def recommend(
        self,
        vibe_query: str,
        top_k: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.25,
        quality_weight: float = 0.10,
        min_rating: float = 0.0,
        min_ratings_count: int = 500,
        required_genres: list[str] | None = None,
        excluded_genres: list[str] | None = None,
    ) -> pd.DataFrame:
        if not vibe_query or not vibe_query.strip():
            return pd.DataFrame()

        expanded_query = self.expand_vibe_query(vibe_query)
        q_emb = self.encode_query(expanded_query)

        # Apply user filters first using fast categorical lookup
        mask = (self.df["Avg_Rating"].values >= min_rating) & (self.num_ratings_arr >= min_ratings_count)
        if required_genres:
            mask &= self._genre_mask(required_genres)
        if excluded_genres:
            mask &= ~self._genre_mask(excluded_genres)

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return pd.DataFrame()

        full_genre_boost = self._fast_category_genre_boost(vibe_query, expanded_query)

        if self.has_aligned_embeddings and self.emb_tensor is not None:
            # Stage 1: Neural + Genre + Quality retrieval across all filtered books
            semantic_scores = self._compute_semantic_scores_chunked(q_emb)
            eff_quality_w = max(quality_weight, 0.30)
            stage1_score = (
                semantic_weight * semantic_scores[filtered_indices]
                + eff_quality_w * self.quality_score[filtered_indices]
                + full_genre_boost[filtered_indices]
            )
            cand_pool_size = min(len(filtered_indices), 1500)
            if len(filtered_indices) > cand_pool_size:
                part_idx = np.argpartition(stage1_score, -cand_pool_size)[-cand_pool_size:]
                pool_indices = filtered_indices[part_idx]
            else:
                pool_indices = filtered_indices

            pool_df = self.df.iloc[pool_indices]
            pool_sem = semantic_scores[pool_indices]

            # Stage 2: On-the-fly TF-IDF lexical scoring on the candidate pool (saves 538 MB RAM!)
            q_tfidf = self.tfidf.transform([vibe_query + " " + expanded_query])
            cand_docs = (
                pool_df["Genres_Clean"].astype(str) + ". " + pool_df["Description"].astype(str)
            ).tolist()
            cand_tfidf = self.tfidf.transform(cand_docs)
            pool_lex_raw = cosine_similarity(cand_tfidf, q_tfidf).flatten().astype(np.float32)
            max_lex = float(pool_lex_raw.max()) if len(pool_lex_raw) > 0 else 0.0
            pool_lex = np.sqrt(pool_lex_raw / max_lex) * 0.70 if max_lex > 0 else pool_lex_raw
        else:
            q_tfidf = self.tfidf.transform([vibe_query + " " + expanded_query])
            lexical_scores = cosine_similarity(self.tfidf_matrix, q_tfidf).flatten().astype(np.float32)
            lexical_norm = (lexical_scores / lexical_scores.max()) if lexical_scores.max() > 0 else lexical_scores
            prelim = lexical_weight * lexical_norm[filtered_indices] + quality_weight * self.quality_score[filtered_indices]
            cand_pool_size = min(len(filtered_indices), 250)
            part_idx = np.argsort(prelim)[::-1][:cand_pool_size]
            pool_indices = filtered_indices[part_idx]
            pool_df = self.df.iloc[pool_indices]
            pool_lex = lexical_norm[pool_indices]
            pool_sem = np.zeros(len(pool_indices), dtype=np.float32)

        # Stage 3: Atmospheric keyword & synopsis quality boost on candidate pool
        q_words = [
            w.lower() for w in re.split(r"[,.;\s]+", vibe_query)
            if len(w) >= 4 and w.lower() not in {"with", "that", "from", "this", "vibe", "book", "like", "very"}
        ]
        pool_boost = full_genre_boost[pool_indices].copy()
        desc_lower = pool_df["Description"].astype(str).str.lower()
        if q_words:
            for w in q_words:
                stem = w[:5] if len(w) >= 6 else w
                in_desc = desc_lower.str.contains(stem, regex=False).values
                pool_boost += in_desc.astype(np.float32) * 0.03
            pool_boost = np.clip(pool_boost, 0.0, 0.24)

        # Penalize stub synopses (< 120 chars), anthologies, non-fiction meta-books, and kids series unless requested
        desc_lens = desc_lower.str.len().values
        short_penalty = np.where(desc_lens < 120, -0.12, 0.0).astype(np.float32)
        q_low = vibe_query.lower()
        if not any(k in q_low for k in ("short stor", "anthology", "collection", "tales", "poe", "poem", "comic", "batman")):
            is_anthology_or_meta = (
                desc_lower.str.contains(
                    "anthology|collection of|stories from the|tales of horror from|sixteen tales|thirteen stories|seven exquisite tales|collecting legends of|three investigators|study guide|cultural history of",
                    regex=True,
                ).values
                | pool_df["Book"].astype(str).str.lower().str.contains(
                    "short stories|other stories|tales of|selected tales|complete poems|great mysteries|gothic tales|batman:|gothic!:",
                    regex=True,
                ).values
            )
            short_penalty += np.where(is_anthology_or_meta, -0.12, 0.0).astype(np.float32)
        if not any(k in q_low for k in ("nonfiction", "non-fiction", "history", "biography", "memoir")):
            is_nonfic = pool_df["Genres_Clean"].astype(str).str.contains("Nonfiction", regex=False).values
            short_penalty += np.where(is_nonfic, -0.12, 0.0).astype(np.float32)

        eff_quality_w = max(quality_weight, 0.32)
        pool_total_score = (
            semantic_weight * pool_sem
            + (lexical_weight * 0.65) * pool_lex
            + eff_quality_w * self.quality_score[pool_indices]
            + pool_boost
            + short_penalty
        )

        target_count = offset + top_k
        sorted_sub_idx = np.argsort(pool_total_score)[::-1][: max(target_count * 4, 200)]
        cand_top_indices = pool_indices[sorted_sub_idx]

        # Deduplicate editions by normalized title AND limit repetition by the same author
        seen_titles = set()
        author_counts = {}
        unique_sub_positions = []
        for pos, idx_val in zip(sorted_sub_idx, cand_top_indices):
            row_item = self.df.iloc[idx_val]
            norm_t = _normalize_title(str(row_item["Book"]))
            author_key = str(row_item["Author"]).strip().lower()
            if not norm_t or norm_t in seen_titles:
                continue
            if author_counts.get(author_key, 0) >= 1:
                continue
            seen_titles.add(norm_t)
            author_counts[author_key] = author_counts.get(author_key, 0) + 1
            unique_sub_positions.append(pos)
            if len(unique_sub_positions) >= target_count:
                break

        if not unique_sub_positions:
            return pd.DataFrame()

        # Wrap around if offset exceeds unique candidates in pool
        effective_offset = offset if offset < len(unique_sub_positions) else (offset % len(unique_sub_positions))
        sliced_positions = unique_sub_positions[effective_offset : effective_offset + top_k]
        unique_sub_positions = np.array(sliced_positions, dtype=int)
        top_indices = pool_indices[unique_sub_positions]

        results = self.df.iloc[top_indices].copy()
        results["Genres_List"] = results["Genres_Clean"].astype(str).fillna("Fiction").str.split(", ")
        results["Match_Score"] = pool_total_score[unique_sub_positions]
        results["Semantic_Score"] = pool_sem[unique_sub_positions]
        results["Lexical_Score"] = pool_lex[unique_sub_positions]
        results["Genre_Boost"] = pool_boost[unique_sub_positions]
        results["Matched_Vibes"] = [
            self.extract_matching_highlights(vibe_query, row)
            for _, row in results.iterrows()
        ]
        return results.reset_index(drop=True)

    def find_similar_books(
        self, book_title: str, top_k: int = 6, offset: int = 0, min_ratings_count: int = 25
    ) -> pd.DataFrame:
        matches = self.df[self.df["Book"].str.lower() == book_title.lower()]
        if matches.empty:
            matches = self.df[self.df["Book"].str.lower().str.contains(re.escape(book_title.lower()), na=False)]
        if matches.empty:
            return pd.DataFrame()
        idx = matches.sort_values("Num_Ratings_Int", ascending=False).index[0]
        if self.has_aligned_embeddings and self.emb_tensor is not None:
            with torch.no_grad():
                target_t = self.emb_tensor[idx].float().numpy()
            sims = self._compute_semantic_scores_chunked(target_t)
        else:
            target_vec = self.tfidf_matrix[idx]
            sims = cosine_similarity(self.tfidf_matrix, target_vec).flatten()
        sims[idx] = -1.0
        if min_ratings_count > 0:
            sims = np.where(self.num_ratings_arr >= min_ratings_count, sims, -1.0)
        target_count = offset + top_k
        top_cand_idx = np.argsort(sims)[::-1][: max(target_count * 4, 150)]
        ref_norm = _normalize_title(str(self.df.iloc[idx]["Book"]))
        seen_titles = {ref_norm}
        chosen_idx = []
        for c_idx in top_cand_idx:
            t_norm = _normalize_title(str(self.df.iloc[c_idx]["Book"]))
            if t_norm and t_norm not in seen_titles:
                seen_titles.add(t_norm)
                chosen_idx.append(c_idx)
                if len(chosen_idx) >= target_count:
                    break
        if not chosen_idx:
            return pd.DataFrame()
        effective_offset = offset if offset < len(chosen_idx) else (offset % len(chosen_idx))
        chosen_idx = chosen_idx[effective_offset : effective_offset + top_k]
        res = self.df.iloc[chosen_idx].copy()
        res["Genres_List"] = res["Genres_Clean"].astype(str).fillna("Fiction").str.split(", ")
        res["Match_Score"] = sims[chosen_idx]
        return res.reset_index(drop=True)
