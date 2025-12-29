import os
import subprocess
import argparse

def get_input(prompt, default=None):
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"{prompt} (Mandatory): ").strip()

def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The PDF file to process")
    args = parser.parse_args()

    input_pdf = args.file
    base_name = os.path.splitext(input_pdf)[0]
    markdown_file = f"{base_name}.md"
    clusters_file = f"{base_name}_clusters.json"
    contradictions_file = f"{base_name}_contradictions.json"

    print(f"--- Starting Pipeline for {input_pdf} ---\n")

    # Step 1: Docling (PDF to Markdown)
    print("Step 1: Converting PDF to Markdown...")
    subprocess.run(["docling", input_pdf], check=True)

    # Step 2: Clustering
    print("\n--- Clustering Configuration ---")
    embed_model = get_input("Embedding model", "nomic-embed-text")
    threshold = get_input("Distance threshold (lower = tighter)", "0.45")

    cluster_cmd = [
        "python", "cluster_sentences.py",
        "--input", markdown_file,
        "--markdown",
        "--output", clusters_file,
        "--model", embed_model,
        "--distance-threshold", threshold
    ]
    subprocess.run(cluster_cmd, check=True)

    # Step 3: Contradictions
    print("\n--- Contradiction Check Configuration ---")
    llm_model = get_input("LLM model for contradiction check (e.g., llama3)")

    check_cmd = [
        "python", "check_contradictions.py",
        "--clusters", clusters_file,
        "--output", contradictions_file,
        "--model", llm_model
    ]
    subprocess.run(check_cmd, check=True)

    print(f"\n--- Pipeline Complete! ---")
    print(f"Results saved to: {contradictions_file}")

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nProcess cancelled by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")