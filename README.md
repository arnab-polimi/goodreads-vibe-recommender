# 📚 VibeRead — Goodreads 10M Vibe Book Recommender

**VibeRead** is a hybrid neural + lexical book recommendation engine built on the **Goodreads 10M Book Dataset**. Describe any mood, aesthetic, atmosphere, or feeling (e.g., *"gothic, horror, mysterious and lonely"* or *"dark academia, secret society, obsessive, autumn rain"*), and discover the books that best match your vibe.

---

## ✨ Features

- **58,200+ Curated Goodreads Catalog**: Filtered and deduplicated from the 10-million-book Goodreads dataset (`bahramjannesarr/goodreads-book-datasets-10m`), enriched with Goodreads shelf tags and atmospheric genre inference.
- **Hybrid Vibe Scoring Engine**:
  1. **Neural Semantic Similarity (`all-MiniLM-L6-v2`)**: Captures deep mood, tone, and conceptual atmosphere.
  2. **Lexical Vibe Matching (`TF-IDF` unigrams & bigrams)**: Matches exact vibe descriptors, literary tropes, and themes.
  3. **Curated Vibe Expansion & Shelf Boost**: Expands atmospheric prompts (gothic, cozy, dark academia, cyberpunk, melancholic, etc.) and boosts direct Goodreads shelf hits.
  4. **Goodreads Quality Weighting**: Balances pure vibe similarity with reader ratings and log-popularity.
- **Interactive Streamlit UI**:
  - Highlighted #1 Best Vibe Match card + ranked recommendations
  - Highlighted vibe keywords inside book synopses
  - Adjustable hybrid weights, minimum Goodreads rating filter, and include/exclude genre filters
  - **"Find Similar by Book Title"** tab to find books with the closest atmospheric DNA

---

## 🚀 Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

3. **CLI Mode (Optional):**
   ```bash
   python cli.py "gothic, horror, mysterious and lonely" --top-k 5
   ```

---

## ⚡ Building Full GPU Embeddings on Google Colab

To precompute the full 58,200-book dense embedding matrix in ~45 seconds on a free T4 GPU:

1. Open [`colab_build_index.ipynb`](colab_build_index.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **T4 GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`) and run the cell.
3. Colab will automatically download `goodreads_index_files.zip`.
4. Either drop `goodreads_index_files.zip` into the project root / `data/` folder (it will auto-extract on launch), or upload it directly in the Streamlit sidebar!

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this repository to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/), click **New app**, select your GitHub repository, and set **Main file path** to `app.py`.
3. Click **Deploy**!
