import argparse
from recommender import VibeRecommender


def main():
    parser = argparse.ArgumentParser(description="Recommend Goodreads books by vibe.")
    parser.add_argument(
        "vibe",
        nargs="?",
        default="gothic, horror, mysterious and lonely",
        help="Vibe description (e.g. 'gothic, horror, mysterious and lonely')",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of books to return")
    args = parser.parse_args()

    engine = VibeRecommender()
    results = engine.recommend(args.vibe, top_k=args.top_k)

    print(f"\n🔮 Vibe Query: \"{args.vibe}\"")
    print("=" * 80)
    for idx, row in results.iterrows():
        rank = idx + 1
        badge = "🏆 BEST MATCH" if rank == 1 else f"#{rank}"
        print(f"{badge}: {row['Book']} by {row['Author']}")
        print(f"   ⭐ Goodreads Rating: {row['Avg_Rating']:.2f} ({int(row['Num_Ratings_Int']):,} ratings)")
        print(f"   🎯 Vibe Match Score: {min(int(round(row['Match_Score'] * 100)), 99)}% (Semantic: {int(round(row['Semantic_Score']*100))}%, Lexical: {int(round(row['Lexical_Score']*100))}%)")
        print(f"   🏷️  Genres: {row['Genres_Clean']}")
        print(f"   ✨ Matched Vibe Elements: {', '.join(row['Matched_Vibes'])}")
        snippet = str(row["Description"])[:320].rsplit(" ", 1)[0] + "..."
        print(f"   📖 Synopsis: {snippet}")
        print("-" * 80)


if __name__ == "__main__":
    main()
