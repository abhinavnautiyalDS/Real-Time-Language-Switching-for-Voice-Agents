from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OLD_MODEL_DIR = ROOT / "models" / "multilingual-bert-tiny-final"
ARTIFACT_DIR = ROOT / "artifacts" / "models"
EXTRACT_ROOT = Path(tempfile.gettempdir()) / "multilingual-voice-language-classifier-models"
MAX_LENGTH = 128

MODEL_OPTIONS = {
    "Old E2 — existing model": {
        "kind": "directory",
        "path": OLD_MODEL_DIR,
        "description": "The model already used by the original Streamlit app.",
    },
    "New E2 — multilingual fine-tune": {
        "kind": "zip",
        "path": ARTIFACT_DIR / "bert_tiny_e2_multilingual.zip",
        "extract_name": "bert_tiny_e2_multilingual",
        "description": "2-layer BERT-Tiny; new multilingual tokenizer and fine-tuning.",
    },
    "New E4 — multilingual fine-tune": {
        "kind": "zip",
        "path": ARTIFACT_DIR / "bert_tiny_e4_multilingual.zip",
        "extract_name": "bert_tiny_e4_multilingual",
        "description": "4-layer BERT-Tiny; new multilingual tokenizer and fine-tuning.",
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


def resolve_model_dir(model_name: str) -> Path:
    """Return the existing model directory or safely extract a selected model ZIP."""
    spec = MODEL_OPTIONS[model_name]
    if spec["kind"] == "directory":
        return Path(spec["path"])

    archive = Path(spec["path"])
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
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
    return rows[0], rows, token_count


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
        help="Old E2 is the existing app model; New E2 and New E4 use the newly uploaded model archives.",
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
            with st.spinner(f"Preparing {selected_model}..."):
                model_dir = resolve_model_dir(selected_model)
                tokenizer, model = load_classifier(str(model_dir))
            with st.spinner("Running model inference on CPU..."):
                top, rows, token_count = predict(sentence, tokenizer, model)

            st.caption(f"Selected model: {selected_model} · Loaded from: {model_dir}")
            st.divider()
            st.subheader("Actual model prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted tag", top["Language tag"])
            c2.metric("Language", top["Language"])
            c3.metric("Top-class confidence", f'{top["Confidence (%)"]:.2f}%')

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
                    "class_probabilities": {
                        row["Language tag"]: row["Probability"] for row in rows
                    },
                })
        except Exception as exc:
            st.error("Could not load the selected model or run inference.")
            st.code(str(exc))
            st.info(
                "For the new E2/E4 options, check that the ZIP archives exist under "
                "artifacts/models and Git LFS downloaded their actual contents. "
                "Also check requirements.txt dependencies."
            )

st.divider()
st.caption(
    "Labels are read from each saved model config. The none class represents other/unsupported "
    "language according to the training labels. Select each model and test identical sentences "
    "to compare outputs."
)
