from pathlib import Path
import tempfile
import time
import zipfile
import urllib.request
import urllib.error

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
EXTRACT_ROOT = Path(tempfile.gettempdir()) / "multilingual-voice-language-classifier-models"
OLD_MODEL_ARCHIVE = EXTRACT_ROOT / "downloads" / "bert_tiny_e2_multilingual_version4.3.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "models"
VERSION44_ARCHIVE = EXTRACT_ROOT / "downloads" / "bert_tiny_e2_multilingual4.4.zip"
VERSION44_MODEL_URL = (
    "https://github.com/abhinavnautiyalDS/"
    "Real-Time-Language-Switching-for-Voice-Agents/raw/refs/heads/realtime-v4.4/"
    "artifacts/version_4_4/archives/bert_tiny_e2_multilingual4.4.zip"
)
MAX_LENGTH = 128

# Version 4 and Version 4.1 models are stored on their respective branches.
VERSION4_MODEL_URL = (
    "https://media.githubusercontent.com/media/abhinavnautiyalDS/"
    "Real-Time-Language-Switching-for-Voice-Agents/version4/"
    "models/bert_tiny_e2_multilingual_version4.zip"
)
VERSION4_CACHE_ARCHIVE = EXTRACT_ROOT / "downloads" / "bert_tiny_e2_multilingual_version4.zip"

VERSION41_MODEL_URL = (
    "https://media.githubusercontent.com/media/abhinavnautiyalDS/"
    "Real-Time-Language-Switching-for-Voice-Agents/version4.1/"
    "models/bert_tiny_e2_multilingual_version4.1.zip"
)
VERSION41_CACHE_ARCHIVE = EXTRACT_ROOT / "downloads" / "bert_tiny_e2_multilingual_version4.1.zip"

# The media.githubusercontent.com URL returned 404 in the deployed environment.
# Use GitHub's standard raw download URL for the 4.3 archive instead.
VERSION43_MODEL_URL = (
    "https://github.com/abhinavnautiyalDS/"
    "Real-Time-Language-Switching-for-Voice-Agents/raw/refs/heads/version4.1/"
    "models/bert_tiny_e2_multilingual_version4.3.zip"
)

MODEL_OPTIONS = {
    "4.3 — latest multilingual model": {
        "kind": "remote_zip",
        "path": OLD_MODEL_ARCHIVE,
        "url": VERSION43_MODEL_URL,
        "extract_name": "model_4_3",
        "description": "Latest multilingual BERT-Tiny E2 checkpoint from the version4.1 branch.",
    },
    "New E2 — multilingual fine-tune": {
        "kind": "zip",
        "path": ARTIFACT_DIR / "bert_tiny_e2_multilingual.zip",
        "extract_name": "bert_tiny_e2_multilingual",
        "description": "2-layer BERT-Tiny; new multilingual tokenizer and fine-tuning.",
    },
    "Version 4.1 — latest multilingual E2": {
        "kind": "remote_zip",
        "path": VERSION41_CACHE_ARCHIVE,
        "url": VERSION41_MODEL_URL,
        "extract_name": "bert_tiny_e2_multilingual_version4_1",
        "description": "Latest Version 4.1 multilingual BERT-Tiny E2 model from the version4.1 branch.",
    },
    "Version 4 — multilingual E2": {
        "kind": "remote_zip",
        "path": VERSION4_CACHE_ARCHIVE,
        "url": VERSION4_MODEL_URL,
        "extract_name": "bert_tiny_e2_multilingual_version4",
        "description": "Version 4 BERT-Tiny E2 model from the version4 branch.",
    },
    "4.4 — multilingual E2": {
        "kind": "remote_zip",
        "path": VERSION44_ARCHIVE,
        "url": VERSION44_MODEL_URL,
        "extract_name": "bert_tiny_e2_multilingual_version4_4",
        "description": "Version 4.4 multilingual BERT-Tiny E2 model hosted on the realtime-v4.4 branch.",
    },
}

LANGUAGE_NAMES = {
    "as-IN": "Assamese", "bn-IN": "Bengali", "en-IN": "English (India)",
    "gu-IN": "Gujarati", "hi-IN": "Hindi", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi",
    "none": "Other / unsupported language", "pa-IN": "Punjabi",
    "ta-IN": "Tamil", "te-IN": "Telugu",
}
SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali / Assamese script", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]


def inspect_script(text: str) -> dict:
    """Count visible script cues; this is not a faithful model explanation."""
    counts = {"Latin": 0, **{name: 0 for name, _, _ in SCRIPT_RANGES}}
    for char in text:
        if not char.isalpha():
            continue
        cp = ord(char)
        if ("A" <= char <= "Z") or ("a" <= char <= "z"):
            counts["Latin"] += 1
            continue
        for name, start, end in SCRIPT_RANGES:
            if start <= cp <= end:
                counts[name] += 1
                break
    return {name: count for name, count in counts.items() if count > 0}


def download_archive(url: str, archive: Path, fallback_url: str | None = None) -> None:
    """Download a model ZIP, retrying a fallback URL when configured."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, archive)
    except Exception as first_error:
        archive.unlink(missing_ok=True)
        if not fallback_url or fallback_url == url:
            raise
        try:
            urllib.request.urlretrieve(fallback_url, archive)
        except Exception:
            archive.unlink(missing_ok=True)
            raise first_error


def resolve_model_dir(model_name: str) -> Path:
    """Return an existing model directory or safely download/extract a selected ZIP."""
    spec = MODEL_OPTIONS[model_name]
    if spec["kind"] == "directory":
        return Path(spec["path"])

    archive = Path(spec["path"])
    if spec["kind"] == "remote_zip" and not archive.exists():
        try:
            download_archive(spec["url"], archive, spec.get("fallback_url"))
        except Exception as exc:
            attempted_url = spec["url"]
            raise RuntimeError(
                f"Could not download the selected model from {attempted_url}: {exc}"
            ) from exc

    if not archive.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive}. Make sure the model ZIP is present in "
            "artifacts/models and Git LFS downloaded the actual file."
        )

    target_root = EXTRACT_ROOT / spec["extract_name"]
    if target_root.exists():
        configs = list(target_root.rglob("config.json"))
        if configs:
            return configs[0].parent

    target_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            root = target_root.resolve()
            for member in zf.infolist():
                destination = (target_root / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError("Unsafe path detected in model ZIP.")
            zf.extractall(target_root)
    except zipfile.BadZipFile as exc:
        if spec["kind"] == "remote_zip":
            archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"{archive.name} is not a readable ZIP. The repository may contain a "
            "Git LFS pointer instead of the downloaded model archive."
        ) from exc

    configs = list(target_root.rglob("config.json"))
    if not configs:
        raise FileNotFoundError(
            f"No config.json found after extracting {archive.name}. "
            "Check the archive's internal folder structure."
        )
    return configs[0].parent


@st.cache_resource(show_spinner="Loading selected fine-tuned BERT model...")
def load_classifier(model_path: str):
    path = Path(model_path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Model files not found at {path}. Check the model path and Git LFS assets."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    prediction_start = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
    prediction_latency_ms = (time.perf_counter() - prediction_start) * 1000
    id2label = model.config.id2label
    rows = []
    for idx, prob in enumerate(probabilities.tolist()):
        label = id2label.get(idx, id2label.get(str(idx), f"class-{idx}"))
        rows.append({
            "Language tag": label,
            "Language": LANGUAGE_NAMES.get(label, label),
            "Confidence (%)": round(prob * 100, 2),
            "Probability": prob,
        })
    rows.sort(key=lambda row: row["Probability"], reverse=True)
    token_count = len(tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    return rows[0], rows, token_count, prediction_latency_ms


st.set_page_config(page_title="Multilingual Voice Language Classifier", page_icon="🌐", layout="wide")
st.title("🌐 Multilingual Voice Language Classifier")
st.write(
    "Choose a model and enter a sentence to compare predictions, softmax confidence, "
    "and scores across all 12 language classes."
)
st.caption("Fine-tuned BERT-Tiny · CPU inference · Maximum 128 tokens")

with st.sidebar:
    st.header("Model selection")
    selected_model = st.selectbox(
        "Choose model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Compare 4.3, new E2, Version 4.1, Version 4, and Version 4.4 models.",
    )
    st.caption(MODEL_OPTIONS[selected_model]["description"])
    st.divider()
    st.header("Model details")
    st.write("**Classes:** 12 language tags")
    st.write("**Device:** CPU")
    st.warning(
        "Softmax confidence is not a guarantee of correctness or a calibrated probability. "
        "Evaluate all models against held-out real call transcripts before production use."
    )

# Interactive turn-buffering / reliability-flow simulator (experimental rules).
st.divider()
st.header("🧪 Turn Buffering Simulator — inspect every calculation")
st.write(
    "Type one ASR turn, optionally add a second compatible turn, and inspect the features, "
    "reliability score, buffering decision, combined text, and actual classifier predictions."
)
with st.expander("Decision logic and limitations", expanded=False):
    st.markdown(
        """
        - This simulates post-ASR turn handling; it does not change STT endpointing.
        - Formula: R = (0.25L + 0.20M + 0.20C + 0.20S) / 0.85 when ASR confidence is unavailable.
        - L: min(word_count / reference_words, 1).
        - M: heuristic meaningful-token ratio blended with a known meaningful-phrase flag.
        - C: top softmax score from the selected model; not necessarily calibrated.
        - S: recognized-script coverage, a lightweight clue—not a language prediction.
        - A reviewed meaningful acknowledgement bypasses buffering. Entity/filler flags are manual hints, not a named-entity detector.
        - Weights and threshold are starting values only; evaluate on real call data before production use.
        """
    )

sim_c1, sim_c2 = st.columns(2)
with sim_c1:
    turn1 = st.text_area("Turn 1 (required)", value="Rahul", key="buffer_turn1", height=90)
with sim_c2:
    turn2 = st.text_area(
        "Turn 2 (optional; same speaker/context)",
        value="mujhe appointment chahiye",
        key="buffer_turn2",
        height=90,
        help="Leave empty to simulate waiting for a second turn or timing out.",
    )

sim_r1, sim_r2, sim_r3 = st.columns(3)
with sim_r1:
    ref_words = st.slider("Reference word count", min_value=2, max_value=10, value=5, key="buffer_ref_words")
with sim_r2:
    threshold = st.slider("Reliability threshold", min_value=0.30, max_value=0.95, value=0.65, step=0.05, key="buffer_threshold")
with sim_r3:
    wait_ms = st.slider("Illustrative wait window (ms)", min_value=100, max_value=1000, value=500, step=50, key="buffer_wait_ms")

sim_r4, sim_r5, sim_r6 = st.columns(3)
with sim_r4:
    meaningful_ack = st.checkbox(
        "Turn 1 is a known meaningful acknowledgement",
        value=False,
        key="buffer_ack",
        help="Example: “हाँ, बिल्कुल”. This rule bypasses length-only buffering.",
    )
with sim_r5:
    entity_hint = st.checkbox("Turn 1 may be a name/company/acronym", value=True, key="buffer_entity")
with sim_r6:
    filler_hint = st.checkbox("Turn 1 may be filler/noise only", value=False, key="buffer_filler")

if st.button("Run full turn-flow simulation", type="primary", key="run_turn_flow"):
    first = " ".join(turn1.split())
    second = " ".join(turn2.split())
    if not first:
        st.error("Enter Turn 1 to run the simulation.")
    else:
        try:
            with st.spinner(f"Loading {selected_model} for real classifier scores..."):
                sim_model_dir = resolve_model_dir(selected_model)
                sim_tokenizer, sim_model = load_classifier(str(sim_model_dir))

            def _script_feature(value):
                cues = inspect_script(value)
                alpha_count = sum(ch.isalpha() for ch in value)
                recognized_count = sum(cues.values())
                coverage = recognized_count / max(alpha_count, 1)
                return min(max(coverage, 0.0), 1.0), cues, alpha_count, recognized_count

            def _feature_row(value):
                tokens = value.split()
                n = len(tokens)
                length = min(n / ref_words, 1.0)
                filler_set = {"umm", "um", "uh", "uhh", "hmm", "hmmm", "erm", "ahh"}
                non_filler = [t for t in tokens if t.lower().strip(".,!?;:") not in filler_set]
                token_ratio = len(non_filler) / max(n, 1)
                known_phrase = 1.0 if value.lower().strip(" .,!?") in {
                    "हाँ", "हां", "हाँ बिल्कुल", "हां बिल्कुल", "बिल्कुल",
                    "hello", "hi", "yes", "okay", "ok", "haan", "haan bilkul",
                } else 0.0
                meaningful = min(max(0.5 * token_ratio + 0.5 * known_phrase, 0.0), 1.0)
                script_score, cues, alpha_count, recognized_count = _script_feature(value)
                top, all_rows, token_count, latency = predict(value, sim_tokenizer, sim_model)
                confidence = top["Probability"]
                reliability = (0.25 * length + 0.20 * meaningful + 0.20 * confidence + 0.20 * script_score) / 0.85
                return {
                    "text": value, "word_count": n, "L": length, "M": meaningful,
                    "C": confidence, "S": script_score, "R": reliability,
                    "script_cues": cues, "alpha_count": alpha_count,
                    "recognized_count": recognized_count, "top": top, "rows": all_rows,
                    "token_count": token_count, "latency": latency,
                    "token_ratio": token_ratio, "known_phrase": known_phrase,
                }

            f1 = _feature_row(first)
            if meaningful_ack:
                decision = "CLASSIFY_NOW"
                hard_reason = "Manual meaningful-acknowledgement override is enabled."
            elif filler_hint:
                decision = "BUFFER_AND_WAIT" if not second else "COMBINE_WITH_NEXT"
                hard_reason = "Manual filler/noise hint is enabled; this is not automatic noise detection."
            elif entity_hint and f1["word_count"] <= 2:
                decision = "BUFFER_AND_WAIT" if not second else "COMBINE_WITH_NEXT"
                hard_reason = "Manual name/company/acronym hint on a short turn: buffer as a cautious heuristic, regardless of the raw score."
            elif f1["R"] < threshold:
                decision = "BUFFER_AND_WAIT" if not second else "COMBINE_WITH_NEXT"
                hard_reason = f"Reliability {f1['R']:.3f} is below threshold {threshold:.2f}."
            else:
                decision = "CLASSIFY_NOW"
                hard_reason = f"Reliability {f1['R']:.3f} meets/exceeds threshold {threshold:.2f}."

            st.subheader("1. Turn 1 feature calculations")
            st.code(
                f"Input: {first}\n"
                f"Word count n = {f1['word_count']}\n"
                f"L = min(n / N_ref, 1) = min({f1['word_count']} / {ref_words}, 1) = {f1['L']:.3f}\n"
                f"Meaningful-token ratio = {f1['token_ratio']:.3f}\n"
                f"Known-phrase match = {f1['known_phrase']:.1f}\n"
                f"M = 0.5 × token_ratio + 0.5 × phrase_match = {f1['M']:.3f}\n"
                f"C = actual BERT-Tiny top softmax score = {f1['C']:.3f}\n"
                f"S = recognized-script coverage = {f1['recognized_count']} / max({f1['alpha_count']}, 1) = {f1['S']:.3f}\n"
                f"ASR confidence = unavailable; weights renormalized by 0.85\n"
                f"R = (0.25L + 0.20M + 0.20C + 0.20S) / 0.85 = {f1['R']:.3f}\n"
                f"Threshold = {threshold:.2f}\n"
                f"Decision = {decision}"
            )
            st.caption(hard_reason)
            st.caption(f"Illustrative buffering window: {wait_ms} ms. This UI simulates the decision; it does not actually wait for a live ASR event.")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Length L", f"{f1['L']:.3f}")
            fc2.metric("Meaning M", f"{f1['M']:.3f}")
            fc3.metric("Model confidence C", f"{f1['C']*100:.2f}%")
            fc4.metric("Reliability R", f"{f1['R']:.3f}")
            if decision == "CLASSIFY_NOW":
                st.success("Flow result: classify Turn 1 immediately.")
            elif second:
                st.warning("Flow result: combine Turn 1 + Turn 2, then classify the combined text.")
            else:
                st.warning("Flow result: buffer Turn 1; no second turn supplied, so timeout would flush Turn 1 alone.")

            final_text = first
            if decision == "COMBINE_WITH_NEXT" and second:
                final_text = first + " " + second
            st.subheader("2. Text sent to the language classifier")
            st.code(final_text)
            if decision == "BUFFER_AND_WAIT" and not second:
                st.info("No Turn 2 was entered. On a real timeout, the buffered Turn 1 would be classified alone.")
            elif decision == "CLASSIFY_NOW":
                st.write("The simulator's selected output is Turn 1. Turn 2 is not needed for this decision.")
            else:
                st.write("The simulator's selected output combines the two supplied turns (same speaker/context assumed).")

            st.subheader("3. Actual BERT-Tiny prediction for selected output")
            chosen_top, chosen_rows, chosen_tokens, chosen_latency = predict(final_text, sim_tokenizer, sim_model)
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Predicted language tag", chosen_top["Language tag"])
            pc2.metric("Language", chosen_top["Language"])
            pc3.metric("Top softmax score", f"{chosen_top['Confidence (%)']:.2f}%")
            st.write(f"Classifier input tokens: {chosen_tokens} · Prediction latency: {chosen_latency:.2f} ms")
            st.dataframe(pd.DataFrame([{
                "Language tag": row["Language tag"],
                "Language": row["Language"],
                "Confidence (%)": row["Confidence (%)"],
            } for row in chosen_rows]), hide_index=True, use_container_width=True)

            st.subheader("4. Compare single-turn vs combined-turn predictions")
            if second:
                raw_top, raw_rows, raw_tokens, raw_latency = predict(first, sim_tokenizer, sim_model)
                combined_text = first + " " + second
                combined_top, combined_rows, combined_tokens, combined_latency = predict(combined_text, sim_tokenizer, sim_model)
                compare = pd.DataFrame([
                    {"Input strategy": "Turn 1 only", "Text": first, "Predicted tag": raw_top["Language tag"],
                     "Top score (%)": raw_top["Confidence (%)"], "Tokens": raw_tokens, "Inference ms": round(raw_latency, 2)},
                    {"Input strategy": "Turn 1 + Turn 2", "Text": combined_text, "Predicted tag": combined_top["Language tag"],
                     "Top score (%)": combined_top["Confidence (%)"], "Tokens": combined_tokens, "Inference ms": round(combined_latency, 2)},
                ])
                st.dataframe(compare, hide_index=True, use_container_width=True)
                st.caption("A higher score does not prove a prediction is more correct without ground-truth labels.")
            st.subheader("5. Script cues (observable, not model explanation)")
            st.json(f1["script_cues"] if f1["script_cues"] else {"message": "No recognized script cues found."})
        except Exception as exc:
            st.error("Could not run the turn-flow simulation.")
            st.code(str(exc))
            st.info("Check that the selected model ZIP is downloadable and contains config.json, tokenizer files, and model weights.")

text_input = st.text_area(
    "Sentence to classify",
    value="Eligibility क्या है?",
    height=120,
    placeholder="Type Hindi, Hinglish, English, Punjabi, or another language...",
)

if st.button("Classify sentence", type="primary"):
    sentence = text_input.strip()
    if not sentence:
        st.error("Please enter a sentence first.")
    else:
        try:
            request_start = time.perf_counter()
            load_start = time.perf_counter()
            with st.spinner(f"Preparing {selected_model}..."):
                model_dir = resolve_model_dir(selected_model)
                tokenizer, model = load_classifier(str(model_dir))
            model_load_latency_ms = (time.perf_counter() - load_start) * 1000

            with st.spinner("Running model inference on CPU..."):
                top, rows, token_count, prediction_latency_ms = predict(sentence, tokenizer, model)
            end_to_end_latency_ms = (time.perf_counter() - request_start) * 1000

            st.caption(f"Selected model: {selected_model} · Loaded from: {model_dir}")
            st.divider()
            st.subheader("Actual model prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted tag", top["Language tag"])
            c2.metric("Language", top["Language"])
            c3.metric("Top-class confidence", f'{top["Confidence (%)"]:.2f}%')

            st.subheader("Latency")
            l1, l2, l3 = st.columns(3)
            l1.metric("Prediction latency", f"{prediction_latency_ms:.2f} ms")
            l2.metric("Model load / setup", f"{model_load_latency_ms:.2f} ms")
            l3.metric("Total request time", f"{end_to_end_latency_ms:.2f} ms")
            st.caption(
                "Prediction latency includes tokenization and the model forward pass. "
                "Model load/setup includes model resolution and loading; cached models make "
                "subsequent runs much faster. Total request time includes both stages."
            )

            if top["Confidence (%)"] >= 80:
                st.success("The model assigns a high softmax score to its top class.")
            elif top["Confidence (%)"] >= 50:
                st.info("The top class leads, but alternatives may still be plausible.")
            else:
                st.warning("The top score is low; interpret this prediction cautiously.")

            st.subheader("Why this prediction?")
            st.write(
                "The returned tag is the class with the highest softmax score from the model's logits. "
                "This classifier does not inherently generate a faithful natural-language explanation. "
                "Script counts below are observable clues in the input, not proof of what caused the model's decision."
            )
            left, right = st.columns(2)
            with left:
                st.write("**Visible script cues**")
                cues = inspect_script(sentence)
                if cues:
                    st.dataframe(
                        pd.DataFrame([
                            {"Script": name, "Alphabetic characters": count}
                            for name, count in cues.items()
                        ]),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.write("No recognized script cues found.")
            with right:
                st.write("**Input processing**")
                st.write(f"- Characters: {len(sentence)}")
                st.write(f"- Tokens including special tokens: {token_count}")
                st.write(f"- Truncation limit: {MAX_LENGTH} tokens")
                st.write("- Device: CPU")

            st.subheader("All class scores")
            score_df = pd.DataFrame([
                {
                    "Language tag": row["Language tag"],
                    "Language": row["Language"],
                    "Confidence (%)": row["Confidence (%)"],
                }
                for row in rows
            ])
            st.bar_chart(score_df.set_index("Language tag")[["Confidence (%)"]])
            st.dataframe(score_df, hide_index=True, use_container_width=True)

            with st.expander("Raw model output"):
                st.json({
                    "selected_model": selected_model,
                    "model_directory": str(model_dir),
                    "predicted_label": top["Language tag"],
                    "predicted_language": top["Language"],
                    "top_softmax_score": top["Probability"],
                    "latency_ms": {
                        "prediction_tokenization_and_forward_pass": round(prediction_latency_ms, 3),
                        "model_load_and_setup": round(model_load_latency_ms, 3),
                        "end_to_end_request": round(end_to_end_latency_ms, 3),
                    },
                    "class_probabilities": {
                        row["Language tag"]: row["Probability"] for row in rows
                    },
                })

        except Exception as exc:
            st.error("Could not load the selected model or run inference.")
            st.code(str(exc))
            st.info(
                "For the new E2 option, check that its ZIP archive exists under "
                "artifacts/models and Git LFS downloaded its actual contents. "
                "For Version 4 and Version 4.1, check network access to the branch model ZIP. "
                "Also check requirements.txt dependencies."
            )


# Evaluation lab: compare all configured models against one labelled sheet.
st.divider()
st.header("📊 Evaluation Lab — compare models on the same evaluation sheet")
st.write(
    "Upload a CSV or Excel file with ground-truth columns named `text` and `label`. "
    "Every selected model predicts the same rows; the UI compares those predictions with the labels."
)
with st.expander("Evaluation file requirements", expanded=False):
    st.markdown(
        "- Required columns: `text` and `label` (labels must match model output tags, e.g. `hi-IN`).\n"
        "- Extra columns such as `case_id`, `language`, `source_type`, and `has_number` are retained in downloads.\n"
        "- Metrics include accuracy, macro/weighted F1, per-class precision/recall/F1 and confusion matrices.\n"
        "- Batch latency is throughput-normalized ms/row, not true single-request p50/p90."
    )

eval_upload = st.file_uploader("Upload evaluation sheet (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"], key="model_eval_upload")
eval_models = st.multiselect("Models to compare", options=list(MODEL_OPTIONS.keys()), default=list(MODEL_OPTIONS.keys()), key="model_eval_select")

def _read_evaluation_sheet(upload):
    frame = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Evaluation sheet must include columns named exactly 'text' and 'label'.")
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-empty text/label rows were found.")
    return frame

def _evaluate_one_model(model_name, frame):
    model_dir = resolve_model_dir(model_name)
    tokenizer, model = load_classifier(str(model_dir))
    texts = frame["text"].tolist()
    predictions, confidences, batch_ms_each = [], [], []
    batch_size = 64
    progress = st.progress(0.0, text=f"Evaluating {model_name}...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        id2label = model.config.id2label or {}
        for row, prob in enumerate(probs):
            idx = int(prob.argmax())
            predictions.append(str(id2label.get(idx, id2label.get(str(idx), f"class-{idx}"))))
            confidences.append(float(prob[idx]))
            batch_ms_each.append(elapsed_ms / max(len(batch), 1))
        progress.progress(min(1.0, (start + len(batch)) / max(len(texts), 1)))
    progress.empty()
    truth = frame["label"].astype(str).tolist()
    labels = sorted(set(truth) | set(predictions))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(truth, predictions, labels=labels, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(truth, predictions, labels=labels, average="weighted", zero_division=0)
    return {
        "model": model_name, "predictions": predictions, "confidence": confidences, "labels": labels,
        "accuracy": accuracy_score(truth, predictions), "macro_precision": macro_p,
        "macro_recall": macro_r, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "p50_batch_ms_row": float(pd.Series(batch_ms_each).quantile(.50)),
        "p90_batch_ms_row": float(pd.Series(batch_ms_each).quantile(.90)),
        "report": classification_report(truth, predictions, labels=labels, output_dict=True, zero_division=0),
        "matrix": confusion_matrix(truth, predictions, labels=labels),
    }

if eval_upload is not None:
    try:
        evaluation_df = _read_evaluation_sheet(eval_upload)
        st.success(f"Loaded {len(evaluation_df):,} labelled examples.")
        st.dataframe(evaluation_df.head(10), use_container_width=True)
        if len(evaluation_df) > 50000:
            st.warning("This is a large evaluation file. Running every model on CPU may take a while.")
        if st.button("Run evaluation for selected models", type="primary", key="run_all_model_eval"):
            if not eval_models:
                st.error("Select at least one model.")
            else:
                completed, failed = [], []
                for model_name in eval_models:
                    st.markdown(f"#### {model_name}")
                    try:
                        with st.spinner(f"Loading and evaluating {model_name}..."):
                            result = _evaluate_one_model(model_name, evaluation_df)
                        completed.append(result)
                        st.success(f"{model_name}: evaluation completed.")
                    except Exception as exc:
                        failed.append({"Model": model_name, "Error": str(exc)})
                        st.error(f"{model_name} failed: {exc}")
                if completed:
                    summary = pd.DataFrame([{
                        "Model": x["model"], "Accuracy": x["accuracy"],
                        "Macro precision": x["macro_precision"], "Macro recall": x["macro_recall"],
                        "Macro F1": x["macro_f1"], "Weighted F1": x["weighted_f1"],
                        "P50 batch ms/row": x["p50_batch_ms_row"],
                        "P90 batch ms/row": x["p90_batch_ms_row"], "Examples": len(evaluation_df),
                    } for x in completed]).sort_values("Macro F1", ascending=False)
                    st.subheader("🏆 Leaderboard")
                    st.dataframe(summary.style.format({
                        "Accuracy": "{:.4f}", "Macro precision": "{:.4f}", "Macro recall": "{:.4f}",
                        "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}",
                        "P50 batch ms/row": "{:.3f}", "P90 batch ms/row": "{:.3f}",
                    }), hide_index=True, use_container_width=True)
                    st.subheader("Model metric comparison")
                    st.bar_chart(summary.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]])
                    best = summary.iloc[0]
                    st.success(f"Best by macro-F1 on this evaluation sheet: {best['Model']} (macro-F1 = {best['Macro F1']:.4f}).")
                    st.caption("This winner applies to this evaluation set only; check class-level recall and confusion matrices before deciding.")
                    for result in completed:
                        with st.expander(f"{result['model']} — per-class metrics and confusion matrix"):
                            report_df = pd.DataFrame(result["report"]).T.reset_index().rename(columns={"index": "Class"})
                            st.dataframe(report_df, use_container_width=True)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ConfusionMatrixDisplay(confusion_matrix=result["matrix"], display_labels=result["labels"]).plot(
                                ax=ax, xticks_rotation=45, values_format="d", colorbar=False
                            )
                            ax.set_title(f"{result['model']} confusion matrix")
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    combined = evaluation_df.copy()
                    for result in completed:
                        key = result["model"].lower().replace(" ", "_").replace(".", "_")
                        combined[f"prediction_{key}"] = result["predictions"]
                        combined[f"confidence_{key}"] = result["confidence"]
                    st.download_button(
                        "Download combined predictions for all models",
                        combined.to_csv(index=False).encode("utf-8-sig"),
                        file_name="model_evaluation_all_predictions.csv",
                        mime="text/csv", key="download_combined_eval",
                    )
                if failed:
                    st.subheader("Models that could not be evaluated")
                    st.dataframe(pd.DataFrame(failed), hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read the evaluation sheet: {exc}")


st.divider()
st.caption(
    "Labels are read from each saved model config. The none class represents other/unsupported "
    "language according to the training labels. Select each model and test identical sentences "
    "to compare outputs."
)
