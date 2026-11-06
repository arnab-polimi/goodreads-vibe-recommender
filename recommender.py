import os
import ast
import re
import pickle
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "goodreads_books.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "book_embeddings.npy")
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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class VibeRecommender:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._maybe_extract_colab_zip()

        if not os.path.exists(CSV_PATH) or not os.path.exists(TFIDF_MATRIX_PATH):
            from build_index import build_data_and_index
            build_data_and_index()

        self.df = pd.read_csv(CSV_PATH, low_memory=False)
        if "Vibe_Document" in self.df.columns:
            self.df = self.df.drop(columns=["Vibe_Document"])
        if "Genres_Clean" in self.df.columns:
            self.df["Genres_List"] = self.df["Genres_Clean"].fillna("Fiction").astype(str).str.split(", ")
        else:
            self.df["Genres_List"] = self.df["Genres"].apply(self._safe_parse_list)
        self.embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r") if os.path.exists(EMBEDDINGS_PATH) else None

        with open(TFIDF_PATH, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        self.has_aligned_embeddings = (
            self.embeddings is not None and self.embeddings.shape[0] == len(self.df)
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        # Precompute normalized log-popularity & rating quality score (0 to 1)
        ratings = self.df["Avg_Rating"].fillna(0.0).values
        num_ratings = np.log1p(self.df["Num_Ratings_Int"].fillna(0).values)
        max_log_ratings = max(num_ratings.max(), 1.0)
        self.quality_score = (ratings / 5.0) * 0.6 + (num_ratings / max_log_ratings) * 0.4

        # Extract all top genres for filtering
        all_genres = {}
        for glist in self.df["Genres_List"]:
            for g in glist:
                all_genres[g] = all_genres.get(g, 0) + 1
        self.popular_genres = [
            g for g, count in sorted(all_genres.items(), key=lambda x: x[1], reverse=True)
            if count >= 50
        ]

    @staticmethod
    def _maybe_extract_colab_zip():
        for zip_path in COLAB_ZIP_CANDIDATES:
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(DATA_DIR)
                    print(f"Extracted index files from {zip_path} into {DATA_DIR}")
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
                expanded_terms.extend(synonyms)
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

    def recommend(
        self,
        vibe_query: str,
        top_k: int = 10,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.25,
        quality_weight: float = 0.10,
        min_rating: float = 0.0,
        required_genres: list[str] | None = None,
        excluded_genres: list[str] | None = None,
    ) -> pd.DataFrame:
        if not vibe_query or not vibe_query.strip():
            return pd.DataFrame()

        expanded_query = self.expand_vibe_query(vibe_query)
        q_emb = self.encode_query(expanded_query)

        # 1. Lexical TF-IDF Similarity (using both raw query and expanded query)
        q_tfidf = self.tfidf.transform([vibe_query + " " + expanded_query])
        lexical_scores = cosine_similarity(self.tfidf_matrix, q_tfidf).flatten()

        # Normalize lexical scores to [0, 1] relative to max hit so it blends smoothly with semantic similarity
        if lexical_scores.max() > 0:
            lexical_norm = lexical_scores / lexical_scores.max()
        else:
            lexical_norm = lexical_scores

        # 2. Genre / Vibe direct overlap boost
        # If the user's vibe words directly match the book's Goodreads shelves/genres, give a strong accuracy boost
        q_words = [
            w.lower() for w in re.split(r"[,.;\s]+", vibe_query)
            if len(w) >= 4 and w.lower() not in {"with", "that", "from", "this", "vibe", "book", "like", "very"}
        ]
        genre_boost = np.zeros(len(self.df), dtype=np.float32)
        if q_words:
            genres_lower = self.df["Genres_Clean"].str.lower().fillna("")
            desc_lower = self.df["Description"].str.lower().fillna("")
            for w in q_words:
                # Root stem prefix (e.g., "mysterious" -> "myster", "lonely" -> "lonel")
                stem = w[:5] if len(w) >= 6 else w
                in_genre = genres_lower.str.contains(stem, regex=False).values
                in_desc = desc_lower.str.contains(stem, regex=False).values
                genre_boost += in_genre.astype(np.float32) * 0.08 + in_desc.astype(np.float32) * 0.04
            genre_boost = np.clip(genre_boost, 0.0, 0.25)

        # 3. Neural Semantic Similarity
        if self.has_aligned_embeddings:
            q_cast = q_emb.astype(self.embeddings.dtype, copy=False)
            semantic_scores = np.matmul(self.embeddings, q_cast).astype(np.float32)
            semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
        else:
            # Fast two-stage neural reranking if full 58k embeddings aren't placed in data/ yet
            semantic_scores = np.zeros(len(self.df), dtype=np.float32)
            prelim_score = 0.65 * lexical_norm + 0.20 * self.quality_score + genre_boost
            top_cand_idx = np.argsort(prelim_score)[::-1][:64]
            cand_rows = self.df.iloc[top_cand_idx]
            cand_texts = (
                "Book Title: " + cand_rows["Book"].astype(str) + " by " + cand_rows["Author"].astype(str) + ". "
                + "Atmosphere, Vibe & Genres: " + cand_rows["Genres_Clean"].astype(str) + ". "
                + "Synopsis: " + cand_rows["Description"].astype(str)
            ).tolist()
            with torch.no_grad():
                enc = self.tokenizer(
                    cand_texts, padding=True, truncation=True, max_length=192, return_tensors="pt"
                ).to(self.device)
                out = self.model(**enc)
                cand_embs = F.normalize(mean_pooling(out, enc["attention_mask"]), p=2, dim=1)
                cand_sims = np.dot(cand_embs.cpu().numpy().astype(np.float32), q_emb)
                semantic_scores[top_cand_idx] = np.clip(cand_sims, 0.0, 1.0)

        # Hybrid score calculation
        total_score = (
            semantic_weight * semantic_scores
            + lexical_weight * lexical_norm
            + quality_weight * self.quality_score
            + genre_boost
        )

        # Apply user filters
        mask = self.df["Avg_Rating"] >= min_rating
        if required_genres:
            req_set = set(required_genres)
            mask &= self.df["Genres_List"].apply(lambda glist: bool(req_set.intersection(glist)))
        if excluded_genres:
            excl_set = set(excluded_genres)
            mask &= ~self.df["Genres_List"].apply(lambda glist: bool(excl_set.intersection(glist)))

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return pd.DataFrame()

        # Sort filtered indices by total_score descending
        sorted_sub_idx = np.argsort(total_score[filtered_indices])[::-1][:top_k]
        top_indices = filtered_indices[sorted_sub_idx]

        results = self.df.iloc[top_indices].copy()
        results["Match_Score"] = total_score[top_indices]
        results["Semantic_Score"] = semantic_scores[top_indices]
        results["Lexical_Score"] = lexical_norm[top_indices]
        results["Genre_Boost"] = genre_boost[top_indices]
        results["Matched_Vibes"] = [
            self.extract_matching_highlights(vibe_query, row)
            for _, row in results.iterrows()
        ]
        return results.reset_index(drop=True)

    def find_similar_books(self, book_title: str, top_k: int = 6) -> pd.DataFrame:
        matches = self.df[self.df["Book"].str.lower() == book_title.lower()]
        if matches.empty:
            return pd.DataFrame()
        idx = matches.index[0]
        if self.has_aligned_embeddings:
            target_emb = self.embeddings[idx]
            sims = np.matmul(self.embeddings, target_emb).astype(np.float32)
        else:
            target_vec = self.tfidf_matrix[idx]
            sims = cosine_similarity(self.tfidf_matrix, target_vec).flatten()
        sims[idx] = -1.0  # exclude self
        top_idx = np.argsort(sims)[::-1][:top_k]
        res = self.df.iloc[top_idx].copy()
        res["Match_Score"] = sims[top_idx]
        return res.reset_index(drop=True)
