import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024
COLLECTION = "esg"
STORE_PATH = "store/qdrant"
BATCH_SIZE = 64
_NAMESPACE = uuid.UUID("e56a0000-0000-0000-0000-000000000000")


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found - run src/ingest.py first.")
    with path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    if not chunks:
        sys.exit(f"{path} is empty.")

    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts. Imported lazily so the Qdrant path can be tested without loading the model."""

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_index(
    chunks: list[dict],
    vectors: np.ndarray,
    store_path: str = STORE_PATH,
    collection: str = COLLECTION,
    recreate: bool = False,
) -> None:

    assert len(chunks) == len(vectors), "chunks and vectors length mismatch"
    Path(store_path).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=store_path)

    exists = client.collection_exists(collection)

    if recreate or not exists:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vectors.shape[1], distance=Distance.COSINE
            ),
        )

    points = [
        PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, c["chunk_id"])),
            vector=vec.tolist(),
            payload=c,
        )
        for c, vec in zip(chunks, vectors)
    ]

    client.upsert(collection_name=collection, points=points)
    print(f"Upserted {len(points)} points into '{collection}' at {store_path}")
    client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--store", default=STORE_PATH)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.chunks))
    vectors = embed([c["text"] for c in chunks])
    build_index(chunks, vectors, args.store, args.collection, args.recreate)


if __name__ == "__main__":
    main()
