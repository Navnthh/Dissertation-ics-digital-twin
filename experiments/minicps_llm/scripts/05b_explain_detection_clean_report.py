import argparse
import json
import re
from pathlib import Path

import chromadb


PROJECT_DIR = Path("/home/nr419/ics_rag_llm_pipeline")
CHROMA_PATH = PROJECT_DIR / "data" / "rag_store_proper" / "chroma_db"
COLLECTION_NAME = "ics_rag_knowledge"

OUTPUT_DIR = PROJECT_DIR / "results" / "rag_explanations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
            "distance": float(distance),
            "doc_type": meta.get("doc_type", ""),
            "source_name": meta.get("source_name", ""),
            "external_id": meta.get("external_id", ""),
            "name": meta.get("name", ""),
            "label_text": meta.get("label_text", ""),
            "source_url": meta.get("source_url", ""),
            "snippet": re.sub(r"\s+", " ", doc[:1000]).strip()
        })

    return rows


def classify_feature_pattern(args):
    notes = []

    if abs(args.s1_change) >= 0.8:
        notes.append(
            "S1 tank-level movement is large within the 30-cycle window, "
            "which makes the window behaviourally important."
        )

    if args.a1_open_ratio >= 0.65:
        notes.append(
            "A1 valve is open for a large portion of the window, meaning the controller "
            "is actively influencing the process while the sensed tank level is changing."
        )

    if args.a1_transitions >= 4:
        notes.append(
            "A1 valve changes state multiple times, so the window contains active process transitions."
        )

    if args.prediction.lower() == "attack":
        notes.append(
            "The fine-tuned detector classified the combined feature pattern as attack-like."
        )
    else:
        notes.append(
            "The fine-tuned detector classified the combined feature pattern as normal-like."
        )

    return notes


def build_clean_explanation(args, dtdl_results, minicps_results):
    feature_notes = classify_feature_pattern(args)

    if args.prediction.lower() == "attack":
        summary = (
            "The fine-tuned LLM classified this MiniCPS window as Attack. "
            "The main evidence is the behaviour of SENSOR1_SENSED, the controller-visible "
            "tank-level signal, together with active valve behaviour from ACTUATOR1. "
            "The S1 value changes across the window while the valve is active for a substantial "
            "portion of the same period. In a cyber-physical control system, this is suspicious "
            "because a slow sensor-drift attack can keep individual readings plausible while "
            "gradually misleading the controller's view of the physical tank state."
        )
    else:
        summary = (
            "The fine-tuned LLM classified this MiniCPS window as Normal. "
            "The feature pattern did not strongly match the slow sensor-drift attack behaviour "
            "learned from the labelled MiniCPS examples. This should still be interpreted with "
            "nearby windows because short windows near transition boundaries can be ambiguous."
        )

    impact = (
        "Potential impact: if SENSOR1_SENSED is manipulated, the controller may make decisions "
        "from a misleading tank-level view. This can affect valve timing, tank refill/drain "
        "behaviour, and the consistency between sensed telemetry and the true process state."
    )

    attack_mapping = {
        "attack_type": "slow sensor drift / telemetry manipulation",
        "primary_signal": "SENSOR1_SENSED",
        "supporting_signal": "ACTUATOR1",
        "ics_concept": "sensor spoofing, telemetry deception, manipulation of view",
        "capec_note": (
            "CAPEC retrieval is not used as an automatic ground-truth mapping because earlier "
            "raw retrieval returned broad or deprecated entries. CAPEC/MITRE mappings should be "
            "reported as curated supporting context rather than direct model output."
        )
    }

    operator_action = (
        "Recommended operator action: inspect the S1 tank-level sensing path, compare "
        "SENSOR1_SENSED with the digital-twin or physical process state, and verify whether "
        "ACTUATOR1 valve behaviour is consistent with expected tank and bottle levels."
    )

    report = {
        "detection": {
            "source_file": args.source_file,
            "cycle_range": args.cycle_range,
            "prediction": args.prediction,
        },
        "feature_summary": {
            "S1_change": args.s1_change,
            "S1_range": args.s1_range,
            "S1_mean": args.s1_mean,
            "S2_change": args.s2_change,
            "S2_range": args.s2_range,
            "S3_change": args.s3_change,
            "S3_range": args.s3_range,
            "A1_open_ratio": args.a1_open_ratio,
            "A1_transitions": args.a1_transitions,
        },
        "clean_explanation": {
            "summary": summary,
            "feature_interpretation": feature_notes,
            "potential_impact": impact,
            "curated_attack_mapping": attack_mapping,
            "operator_action": operator_action,
        },
        "retrieved_context": {
            "digital_twin_context": dtdl_results,
            "similar_minicps_training_windows": minicps_results,
        }
    }

    return report


def write_text_report(report, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("FINE-TUNED LLM + RAG CLEAN EXPLANATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        det = report["detection"]
        feat = report["feature_summary"]
        exp = report["clean_explanation"]

        f.write("DETECTION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Source file: {det['source_file']}\n")
        f.write(f"Cycle range: {det['cycle_range']}\n")
        f.write(f"Prediction: {det['prediction']}\n\n")

        f.write("FEATURE SUMMARY\n")
        f.write("-" * 100 + "\n")
        for key, value in feat.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("RAG-ASSISTED EXPLANATION\n")
        f.write("-" * 100 + "\n")
        f.write(exp["summary"] + "\n\n")

        f.write("Feature interpretation:\n")
        for note in exp["feature_interpretation"]:
            f.write(f"- {note}\n")
        f.write("\n")

        f.write("Potential impact:\n")
        f.write(exp["potential_impact"] + "\n\n")

        f.write("Curated attack mapping:\n")
        mapping = exp["curated_attack_mapping"]
        for key, value in mapping.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("Operator action:\n")
        f.write(exp["operator_action"] + "\n\n")

        f.write("=" * 100 + "\n")
        f.write("RETRIEVED DIGITAL TWIN CONTEXT\n")
        f.write("=" * 100 + "\n")
        for i, row in enumerate(report["retrieved_context"]["digital_twin_context"], start=1):
            f.write(f"\n[{i}] {row['name']} | distance={row['distance']}\n")
            f.write(row["snippet"] + "\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("SIMILAR MINICPS TRAINING WINDOWS\n")
        f.write("=" * 100 + "\n")
        for i, row in enumerate(report["retrieved_context"]["similar_minicps_training_windows"], start=1):
            f.write(
                f"\n[{i}] {row['name']} | label={row['label_text']} | "
                f"distance={row['distance']}\n"
            )
            f.write(row["snippet"] + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--cycle-range", required=True)

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
        f"MiniCPS slow sensor drift attack SENSOR1_SENSED tank level ACTUATOR1 valve. "
        f"Prediction {args.prediction}. "
        f"S1_change {args.s1_change}, S1_range {args.s1_range}, "
        f"A1_open_ratio {args.a1_open_ratio}, A1_transitions {args.a1_transitions}."
    )

    dtdl_results = compact_results(
        query_collection(
            collection,
            query,
            top_k=1,
            where={"doc_type": "local_dtdl_model"}
        )
    )

    minicps_results = compact_results(
        query_collection(
            collection,
            query,
            top_k=args.top_k,
            where={"doc_type": "minicps_training_example"}
        )
    )

    report = build_clean_explanation(args, dtdl_results, minicps_results)

    safe_source = args.source_file.replace(".csv", "").replace("/", "_")
    safe_cycles = args.cycle_range.replace(" ", "").replace("to", "_")

    json_path = OUTPUT_DIR / f"clean_rag_explanation_{safe_source}_{safe_cycles}.json"
    txt_path = OUTPUT_DIR / f"clean_rag_explanation_{safe_source}_{safe_cycles}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_text_report(report, txt_path)

    print("=" * 100)
    print("CLEAN RAG EXPLANATION CREATED")
    print("=" * 100)
    print("JSON:", json_path)
    print("TXT:", txt_path)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(report["clean_explanation"]["summary"])

    print("\nFeature interpretation:")
    for note in report["clean_explanation"]["feature_interpretation"]:
        print("-", note)

    print("\nOperator action:")
    print(report["clean_explanation"]["operator_action"])


if __name__ == "__main__":
    main()
