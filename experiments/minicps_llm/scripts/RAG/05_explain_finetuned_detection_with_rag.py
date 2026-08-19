import argparse
import json
from pathlib import Path

import chromadb


PROJECT_DIR = Path("/home/nr419/ics_rag_llm_pipeline")
CHROMA_PATH = PROJECT_DIR / "data" / "rag_store_proper" / "chroma_db"
COLLECTION_NAME = "ics_rag_knowledge"


def query_collection(collection, query, top_k=3, where=None):
    if where:
        return collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )

    return collection.query(
        query_texts=[query],
        n_results=top_k
    )


def compact_results(results):
    rows = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(docs, metas, distances):
        rows.append({
            "distance": distance,
            "doc_type": meta.get("doc_type", ""),
            "source_name": meta.get("source_name", ""),
            "external_id": meta.get("external_id", ""),
            "name": meta.get("name", ""),
            "label_text": meta.get("label_text", ""),
            "source_url": meta.get("source_url", ""),
            "snippet": doc[:900].replace("\n", " ")
        })

    return rows


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction", default="Attack")
    parser.add_argument("--source-file", default="unknown")
    parser.add_argument("--cycle-range", default="unknown")

    parser.add_argument("--s1-change", type=float, required=True)
    parser.add_argument("--s1-range", type=float, required=True)
    parser.add_argument("--s1-mean", type=float, default=0.0)

    parser.add_argument("--s2-change", type=float, default=0.0)
    parser.add_argument("--s2-range", type=float, default=0.0)

    parser.add_argument("--s3-change", type=float, default=0.0)
    parser.add_argument("--s3-range", type=float, default=0.0)

    parser.add_argument("--a1-open-ratio", type=float, required=True)
    parser.add_argument("--a1-transitions", type=int, required=True)

    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    query = (
        f"MiniCPS ICS sensor drift attack S1 tank sensed level manipulation. "
        f"Prediction {args.prediction}. "
        f"S1 change {args.s1_change}, S1 range {args.s1_range}, "
        f"S2 change {args.s2_change}, S2 range {args.s2_range}, "
        f"S3 change {args.s3_change}, S3 range {args.s3_range}, "
        f"valve open ratio {args.a1_open_ratio}, valve transitions {args.a1_transitions}. "
        f"Explain sensor spoofing, telemetry deception, actuator response, and cyber physical impact."
    )

    dtdl_results = compact_results(
        query_collection(
            collection,
            query,
            1,
            where={"doc_type": "local_dtdl_model"}
        )
    )

    mitre_results = compact_results(
        query_collection(
            collection,
            query,
            args.top_k,
            where={"doc_type": "mitre_attack_ics"}
        )
    )

    capec_results = compact_results(
        query_collection(
            collection,
            query,
            args.top_k,
            where={"doc_type": "capec"}
        )
    )

    print_section("FINE-TUNED LLM DETECTION")
    print(f"Source file: {args.source_file}")
    print(f"Cycle range: {args.cycle_range}")
    print(f"Prediction: {args.prediction}")

    print("\nFeature summary:")
    print(f"- S1 tank sensed change: {args.s1_change}")
    print(f"- S1 tank sensed range: {args.s1_range}")
    print(f"- S1 tank sensed mean: {args.s1_mean}")
    print(f"- S2 flow sensed change: {args.s2_change}")
    print(f"- S2 flow sensed range: {args.s2_range}")
    print(f"- S3 bottle sensed change: {args.s3_change}")
    print(f"- S3 bottle sensed range: {args.s3_range}")
    print(f"- A1 valve open ratio: {args.a1_open_ratio}")
    print(f"- A1 valve transitions: {args.a1_transitions}")

    print_section("RAG-ASSISTED EXPLANATION")

    if args.prediction.lower() == "attack":
        print(
            "The fine-tuned detector classified this window as Attack. "
            "The window should be interpreted as suspicious because the controller-visible "
            "S1 tank level changes over the window while the actuator remains active for a "
            "large portion of the same period. In the MiniCPS process, S1 represents the "
            "tank level seen by the controller, so gradual manipulation of this signal can "
            "create a plausible but misleading process view. This resembles a slow sensor "
            "drift or telemetry deception attack rather than a simple threshold violation."
        )
    else:
        print(
            "The fine-tuned detector classified this window as Normal. "
            "The feature pattern does not strongly match the attack behaviour learned from "
            "the labelled MiniCPS examples, although it should still be interpreted in the "
            "context of nearby windows."
        )

    print_section("RETRIEVED DIGITAL TWIN CONTEXT")
    for i, row in enumerate(dtdl_results, start=1):
        print(f"\n[{i}] {row['name']}")
        print(row["snippet"])

    print_section("MITRE ICS CONTEXT")
    for i, row in enumerate(mitre_results, start=1):
        print(f"\n[{i}] {row['external_id']} {row['name']} | distance={row['distance']}")
        print(row["snippet"])

    print_section("CAPEC CONTEXT")
    print(
        "Note: CAPEC retrieval may include broad or deprecated software attack patterns. "
        "Use these as supporting context only, not as automatic ground-truth mappings."
    )

    for i, row in enumerate(capec_results, start=1):
        print(f"\n[{i}] {row['external_id']} {row['name']} | distance={row['distance']}")
        print(row["snippet"])

    print_section("OPERATOR-READABLE ALERT")
    print(f"Alert: {args.prediction}")
    print(
        "Recommended action: inspect the S1 tank-level sensing path, compare the sensed "
        "tank level against the digital-twin or physical process state, and verify whether "
        "A1 valve activity is consistent with the expected tank and bottle levels."
    )


if __name__ == "__main__":
    main()
