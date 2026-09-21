import argparse
import os

from extract_text import process_folder
from embedding import embed_chunks, model
from hnsw import insert, search_layer
from persistance import save_index, load_index

INDEX_PATH = "index.pkl"
PDFS_FOLDER = "pdfs"


def get_folder_fingerprint(folder_path):
    latest = 0
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        mtime = os.path.getmtime(path)
        if mtime > latest:
            latest = mtime
    return latest


def build_index():
    print("Building index from scratch...")

    chunks = process_folder(PDFS_FOLDER)
    print(f"Found {len(chunks)} chunks")

    chunks = embed_chunks(chunks)

    vectors = []
    graph = []
    node_levels = []
    state = {"entry_point": None, "max_level": -1}

    for chunk in chunks:
        insert(chunk["embedding"], vectors, graph, node_levels, state)

    print(f"Index built with {len(vectors)} nodes")

    fingerprint = get_folder_fingerprint(PDFS_FOLDER)
    save_index(INDEX_PATH, vectors, graph, node_levels, state, chunks, fingerprint)
    print(f"Index saved to {INDEX_PATH}")

    return vectors, graph, node_levels, state, chunks


def load_or_build(force_rebuild=False):
    if force_rebuild and os.path.exists(INDEX_PATH):
        print("--rebuild flag set, removing old index...")
        os.remove(INDEX_PATH)

    current_fingerprint = get_folder_fingerprint(PDFS_FOLDER)

    if os.path.exists(INDEX_PATH):
        vectors, graph, node_levels, state, chunks, saved_fingerprint = load_index(INDEX_PATH)

        if saved_fingerprint == current_fingerprint:
            print(f"Index up to date, loaded {len(vectors)} nodes")
            return vectors, graph, node_levels, state, chunks

        print("PDFs folder changed, rebuilding index...")

    return build_index()


def search(query, model, vectors, graph, state, chunks, k=5):
    # Must match embed_chunks: normalize so euclidean distance in hnsw.py
    # ranks results the same way cosine similarity would.
    query_vector = model.encode(query, normalize_embeddings=True)

    entry_point = state["entry_point"]
    L = state["max_level"]

    for layer in range(L, 0, -1):
        nearest = search_layer(vectors, graph, entry_point, query_vector, 1, layer)
        entry_point = nearest[0]

    results = search_layer(vectors, graph, entry_point, query_vector, max(k, 50), 0)

    top_k = results[:k]
    return [chunks[node_id] for node_id in top_k]


def run_cli(model, vectors, graph, state, chunks, k=5):
    print("Semantic search ready. Type your query (or 'quit' to exit).")

    while True:
        query = input("\n> ").strip()

        if query.lower() in ("quit", "exit"):
            break

        if not query:
            continue

        results = search(query, model, vectors, graph, state, chunks, k=k)

        print(f"\nTop {len(results)} results:\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. [{chunk['source']}]")
            print(f"   {chunk['text'][:200]}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local semantic search over your PDFs")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index from scratch")
    parser.add_argument("-k", type=int, default=5, help="Number of results to return per query")
    args = parser.parse_args()

    vectors, graph, node_levels, state, chunks = load_or_build(force_rebuild=args.rebuild)

    run_cli(model, vectors, graph, state, chunks, k=args.k)