import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import html
import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re


def read_sentences(path: str, markdown: bool = False) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if markdown:
        return list(extract_sentences_from_markdown(text))
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_sentences_from_markdown(text: str) -> Iterable[str]:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if set(line) <= {"|", "-", " ", ":"}:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if cells:
                cleaned_lines.append(", ".join(cells))
            continue
        line = re.sub(r"^[\\-\\*\\+\\d\\.\\)\\s]+", "", line)
        line = html.unescape(line)
        line = line.strip("-* ")
        if line:
            cleaned_lines.append(line)

    for line in cleaned_lines:
        parts = re.split(r"(?<=[.!?;])\\s+(?=[A-Z0-9])", line)
        for part in parts:
            part = part.strip()
            if part:
                yield part


def embed_sentences(sentences: Sequence[str], model: str, url: str) -> np.ndarray:
    embeddings = []
    for sentence in sentences:
        payload = {"model": model, "prompt": sentence}
        resp = requests.post(url, json=payload, timeout=60)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(
                f"Ollama embeddings failed with {resp.status_code}: {resp.text.strip()}"
            ) from None
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        embeddings.append(data["embedding"])
    return np.array(embeddings, dtype=np.float32)


def cluster_embeddings(vectors: np.ndarray, distance_threshold: float) -> np.ndarray:
    if len(vectors) == 0:
        return np.array([])
    # Agglomerative with cosine distance using a precomputed matrix.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    normalized = vectors / norms
    dist_matrix = cosine_distances(normalized)
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels


def name_clusters(sentences: Sequence[str], labels: np.ndarray, top_k: int = 3) -> Dict[int, str]:
    label_to_sents: Dict[int, List[str]] = defaultdict(list)
    for sent, label in zip(sentences, labels):
        label_to_sents[label].append(sent)

    cluster_names: Dict[int, str] = {}
    for label, sents in label_to_sents.items():
        vec = TfidfVectorizer(max_features=32, ngram_range=(1, 2))
        X = vec.fit_transform(sents)
        scores = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        top_terms = vocab[scores.argsort()[::-1][:top_k]]
        cluster_names[label] = ", ".join(top_terms) if len(top_terms) else f"cluster_{label}"
    return cluster_names


def save_clusters(
    sentences: Sequence[str],
    labels: np.ndarray,
    names: Dict[int, str],
    out_path: str,
) -> None:
    data = defaultdict(list)
    for sent, label in zip(sentences, labels):
        data[label].append(sent)

    output = []
    for label, sents in sorted(data.items(), key=lambda kv: kv[0]):
        output.append(
            {
                "cluster_id": int(label),
                "name": names.get(label, f"cluster_{label}"),
                "size": len(sents),
                "sentences": sents,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster sentences using local embeddings from an Ollama model."
    )
    parser.add_argument("--input", default="sentences.txt", help="Input file (sentences per line or markdown).")
    parser.add_argument(
        "--output",
        default="clusters.json",
        help="Where to write clusters as JSON (default: clusters.json).",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model to use (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434/api/embeddings",
        help="Ollama embeddings endpoint.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.35,
        help="Cosine distance threshold for clustering (lower = tighter clusters).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Set if the input is markdown; basic parsing will extract sentences.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = read_sentences(args.input, markdown=args.markdown)
    if not sentences:
        raise SystemExit(f"No sentences found in {args.input}")

    print(f"Embedding {len(sentences)} sentences with model {args.model}...")
    vectors = embed_sentences(sentences, model=args.model, url=args.ollama_url)

    print("Clustering embeddings...")
    labels = cluster_embeddings(vectors, distance_threshold=args.distance_threshold)
    names = name_clusters(sentences, labels)

    save_clusters(sentences, labels, names, args.output)
    print(f"Wrote clusters to {args.output}")


if __name__ == "__main__":
    main()
