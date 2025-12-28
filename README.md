# PDF → Markdown → Clusters → Contradictions

This repo provides a local pipeline to turn PDFs into sentences, cluster them, and flag potential contradictions.

## Prerequisites
- Python 3.8+ with `requests`, `scikit-learn`, `numpy` (already present here).
- `pdftotext` (poppler-utils) if you use `pdf_sentence_parser.py`.
- [Ollama](https://ollama.com/) running locally with:
  - A text LLM (for sentence cleaning and contradiction checks).
  - An embedding model (e.g., `nomic-embed-text`) for clustering.
- Docling (or any PDF→Markdown tool) if you prefer that route.

## Pipeline

### PDF to Markdown (Docling), then cluster
```bash
docling input.pdf        # produces input.md
ollama serve
python cluster_sentences.py --input input.md --markdown --output clusters.json --model nomic-embed-text --distance-threshold 0.45
```
Outputs `clusters.json` with cluster names and member sentences. Raise `--distance-threshold` for fewer/broader clusters; lower it for tighter clusters.

### Contradiction checks
```bash
python check_contradictions.py --clusters clusters.json --output contradictions.json --model <llm-name>
```
Outputs `contradictions.json` with sentence pairs, numeric conflict hints, and an NLI-style verdict (CONTRADICT/AGREE/UNRELATED).

## Notes
- Keep Ollama running during clustering/contradiction steps.
- Use smaller models if you hit memory limits.
- If you switch parsing tools, ensure the input to `cluster_sentences.py` is either one-sentence-per-line text or Markdown with `--markdown`.
