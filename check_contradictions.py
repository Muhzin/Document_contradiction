import argparse
import json
import re
from typing import Dict, List, Tuple

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NUM_UNIT_PATTERN = re.compile(
    r"(?P<num>(?:\d+(?:\.\d+)?))\s*(?P<unit>barg|bar|psi|°c|degc|c|f|°f|week[s]?|day[s]?|%|\$)",
    re.IGNORECASE,
)


def load_clusters(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_numbers_units(text: str) -> List[Tuple[float, str]]:
    matches = []
    for m in NUM_UNIT_PATTERN.finditer(text):
        try:
            val = float(m.group("num"))
        except ValueError:
            continue
        unit = m.group("unit").lower()
        matches.append((val, unit))
    return matches


def likely_conflict_pair(a: str, b: str) -> bool:
    nums_a = extract_numbers_units(a)
    nums_b = extract_numbers_units(b)
    if nums_a and nums_b:
        units_a = {u for _, u in nums_a}
        units_b = {u for _, u in nums_b}
        if units_a & units_b:
            return True
    keywords = ("pressure", "temperature", "week", "day", "percent", "%", "barg", "bar", "psi", "$")
    return any(k in a.lower() and k in b.lower() for k in keywords)


def semantic_dedupe(sentences: List[str], threshold: float = 0.9) -> List[int]:
    vec = TfidfVectorizer().fit(sentences)
    X = vec.transform(sentences)
    sim = cosine_similarity(X)
    keep = []
    for i in range(len(sentences)):
        if any(sim[i, j] >= threshold for j in keep):
            continue
        keep.append(i)
    return keep


def call_nli(ollama_url: str, model: str, premise: str, hypothesis: str) -> str:
    prompt = (
        "Classify the relationship between these statements as one of: CONTRADICT, AGREE, UNRELATED.\n"
        f"Statement A: {premise}\n"
        f"Statement B: {hypothesis}\n"
        "Answer with a single word: CONTRADICT, AGREE, or UNRELATED."
    )
    resp = requests.post(
        ollama_url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip().upper()
    if "CONTRADICT" in text:
        return "CONTRADICT"
    if "AGREE" in text:
        return "AGREE"
    if "UNRELATED" in text:
        return "UNRELATED"
    return text or "UNKNOWN"


def check_cluster(cluster: Dict, model: str, ollama_url: str) -> List[Dict]:
    sentences = cluster.get("sentences", [])
    if len(sentences) < 2:
        return []

    keep_idx = semantic_dedupe(sentences, threshold=0.9)
    filtered = [sentences[i] for i in keep_idx]

    findings = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            a, b = filtered[i], filtered[j]
            if not likely_conflict_pair(a, b):
                continue
            nums_a = extract_numbers_units(a)
            nums_b = extract_numbers_units(b)
            numeric_conflict = False
            detail = ""
            units_intersection = {u for _, u in nums_a} & {u for _, u in nums_b}
            if units_intersection:
                # Compare first matching unit values.
                for ua in units_intersection:
                    vals_a = {v for v, u in nums_a if u == ua}
                    vals_b = {v for v, u in nums_b if u == ua}
                    if vals_a and vals_b and vals_a != vals_b:
                        numeric_conflict = True
                        detail = f"mismatched values for unit {ua}: {sorted(vals_a)} vs {sorted(vals_b)}"
                        break

            verdict = call_nli(ollama_url, model, a, b)
            findings.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "cluster_name": cluster.get("name"),
                    "sentence_a": a,
                    "sentence_b": b,
                    "numeric_conflict": numeric_conflict,
                    "details": detail,
                    "nli": verdict,
                }
            )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect potential contradictions within clusters.")
    parser.add_argument("--clusters", default="clusters.json", help="Input clusters JSON.")
    parser.add_argument("--output", default="contradictions.json", help="Output JSON with findings.")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model for NLI-style prompt.")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama generate endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clusters = load_clusters(args.clusters)
    all_findings: List[Dict] = []
    for cluster in clusters:
        all_findings.extend(check_cluster(cluster, model=args.model, ollama_url=args.ollama_url))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_findings, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(all_findings)} findings to {args.output}")


if __name__ == "__main__":
    main()
