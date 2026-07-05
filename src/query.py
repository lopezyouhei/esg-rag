import argparse
import sys

from qdrant_client import QdrantClient

MODEL_NAME = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
COLLECTION = "esg"
STORE_PATH = "store/qdrant"


def embed_query(text: str) -> list[float]:

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    vector = model.encode([text], normalize_embeddings=True)[0]

    return vector.tolist()


def retrieve(client: QdrantClient, query_vec: list[float], top_k: int) -> list[dict]:
    """Vector search Qdrant and returns payload with the similarity score attached."""
    res = client.query_points(
        collection_name=COLLECTION, query=query_vec, limit=top_k, with_payload=True
    )
    candidates = []
    for pt in res.points:
        payload = dict(pt.payload)
        payload["vector_score"] = pt.score
        candidates.append(payload)

    return candidates


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Reorder candidates with a cross-encoder that reads query + passage together."""

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANKER_NAME)
    scores = reranker.predict([[query, c["text"]] for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument(
        "--top-k", type=int, default=20, help="candidates from vector search"
    )
    ap.add_argument(
        "--top-n", type=int, default=5, help="candidates kept after reranking"
    )
    ap.add_argument("--store", default=STORE_PATH)
    args = ap.parse_args()

    query_vec = embed_query(args.question)
    client = QdrantClient(path=args.store)

    try:
        candidates = retrieve(client, query_vec, args.top_k)
    finally:
        client.close()

    if not candidates:
        sys.exit(
            "No candidates found, is the collection populated? Try running src/index.py"
        )

    ranked = rerank(args.question, candidates, args.top_n)

    print(f"\nQ: {args.question}\n")
    for i, c in enumerate(ranked, 1):
        loc = f"{c['source_file']} p.{c['page_number']}"
        snippet = " ".join(c["text"].split())[:200]
        print(
            f"[{i}] rerank={c['rerank_score']:.3f}  vec={c['vector_score']:.3f}  {loc}"
        )
        print(f"    {c['chunk_id']}")
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    main()
