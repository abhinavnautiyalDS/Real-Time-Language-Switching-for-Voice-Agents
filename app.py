from pathlib import Path
import tempfile, time, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir()) / "rt_lang_models"
MAX_LENGTH, BATCH_SIZE = 128, 64
LANGS = {"as-IN":"Assamese","bn-IN":"Bengali","en-IN":"English","gu-IN":"Gujarati","hi-IN":"Hindi","kn-IN":"Kannada","ml-IN":"Malayalam","mr-IN":"Marathi","none":"Other / unsupported","pa-IN":"Punjabi","ta-IN":"Tamil","te-IN":"Telugu"}
MODELS = {
 "Old E2": {"kind":"dir","path":ROOT/"models/multilingual-bert-tiny-final"},
 "New E2": {"kind":"zip","path":ROOT/"artifacts/models/bert_tiny_e2_multilingual.zip","name":"new_e2"},
 "New E4": {"kind":"zip","path":ROOT/"artifacts/models/bert_tiny_e4_multilingual.zip","name":"new_e4"},
 "Version 4.4": {"kind":"zip","path":ROOT/"artifacts/version_4_4/archives/bert_tiny_e2_multilingual4.4.zip","name":"version_4_4_e2"},
}

def model_dir(name):
    spec=MODELS[name]
    if spec["kind"]=="dir":
        if not (spec["path"]/"config.json").is_file(): raise FileNotFoundError(f"Model config missing: {spec['path']}")
        return spec["path"]
    archive=spec["path"]
    if not archive.is_file(): raise FileNotFoundError(f"Model ZIP missing: {archive}")
    target=TMP/spec["name"]
    configs=list(target.rglob("config.json")) if target.exists() else []
    if configs: return configs[0].parent
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        root=target.resolve()
        for m in z.infolist():
            dest=(target/m.filename).resolve()
            if dest!=root and root not in dest.parents: raise ValueError("Unsafe path in model ZIP")
        z.extractall(target)
    configs=list(target.rglob("config.json"))
    if not configs: raise FileNotFoundError(f"No config.json found inside {archive.name}")
    return configs[0].parent

@st.cache_resource(show_spinner=False)
def load_model(name):
    path=model_dir(name)
    tok=AutoTokenizer.from_pretrained(str(path),use_fast=True,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True)
    model.to("cpu"); model.eval()
    return tok,model,str(path)

def label_id(model,i):
    d=model.config.id2label or {}
    return str(d.get(i,d.get(str(i),f"class-{i}")))

def predict_one(text,tok,model):
    x=tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)
    with torch.inference_mode(): p=torch.softmax(model(**x).logits[0],dim=-1).numpy()
    return label_id(model,int(np.argmax(p))),float(np.max(p)),{label_id(model,i):float(v) for i,v in enumerate(p)}

def read_eval(upload):
    df=pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    df.columns=[str(c).strip() for c in df.columns]
    if not {"text","label"}.issubset(df.columns): raise ValueError("File must have columns named text and label.")
    df=df.dropna(subset=["text","label"]).copy()
    df["text"]=df["text"].astype(str).str.strip(); df["label"]=df["label"].astype(str).str.strip()
    df=df[(df.text!="")&(df.label!="")].reset_index(drop=True)
    if df.empty: raise ValueError("No non-empty labeled rows found.")
    return df

def run_predictions(texts,tok,model):
    preds,confs,perrow=[] ,[],[]
    prog=st.progress(0)
    for start in range(0,len(texts),BATCH_SIZE):
        batch=[str(x) for x in texts[start:start+BATCH_SIZE]]
        enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=MAX_LENGTH)
        t=time.perf_counter()
        with torch.inference_mode(): probs=torch.softmax(model(**enc).logits,dim=-1).numpy()
        ms=(time.perf_counter()-t)*1000
        for row,p in enumerate(probs):
            idx=int(np.argmax(p)); preds.append(label_id(model,idx)); confs.append(float(p[idx])); perrow.append(ms/len(batch))
        prog.progress(min(1.0,(start+len(batch))/len(texts)))
    prog.empty()
    return preds,confs,perrow

def evaluate(name,df):
    tok,model,path=load_model(name)
    preds,conf,lat=run_predictions(df.text.tolist(),tok,model)
    truth=df.label.tolist(); labels=sorted(set(truth)|set(preds))
    p,r,f,_=precision_recall_fscore_support(truth,preds,labels=labels,average="macro",zero_division=0)
    _,_,wf,_=precision_recall_fscore_support(truth,preds,labels=labels,average="weighted",zero_division=0)
    return {"name":name,"path":path,"preds":preds,"confidence":conf,"truth":truth,"labels":labels,
      "accuracy":accuracy_score(truth,preds),"macro_precision":p,"macro_recall":r,"macro_f1":f,"weighted_f1":wf,
      "p50_ms_row":float(np.percentile(lat,50)),"p90_ms_row":float(np.percentile(lat,90)),
      "report":classification_report(truth,preds,labels=labels,output_dict=True,zero_division=0),
      "cm":confusion_matrix(truth,preds,labels=labels)}

st.set_page_config(page_title="Real-Time Language Switching Model Lab",page_icon="🌐",layout="wide")
st.title("🌐 Real-Time Language Switching — Model Lab")
st.write("Run single-text predictions or compare every model on the same labeled evaluation sheet.")
st.caption("CPU inference · Version 4.4 included · Maximum sequence length 128")
single,bench=st.tabs(["Single prediction","Evaluation and model comparison"])
with st.sidebar:
    st.header("Models")
    for n in MODELS: st.write(n)
    st.caption("Evaluation metrics depend on your uploaded ground-truth labels. Softmax is not calibrated confidence.")

with single:
    chosen=st.selectbox("Choose model",list(MODELS),key="single_model")
    sentence=st.text_area("Text to classify",value="Eligibility क्या है?")
    if st.button("Classify",type="primary"):
        if not sentence.strip(): st.warning("Enter text first.")
        else:
            try:
                with st.spinner("Loading model..."): tok,model,path=load_model(chosen)
                t=time.perf_counter(); label,conf,scores=predict_one(sentence.strip(),tok,model); elapsed=(time.perf_counter()-t)*1000
                a,b,c=st.columns(3); a.metric("Predicted tag",label); b.metric("Language",LANGS.get(label,label)); c.metric("Inference",f"{elapsed:.2f} ms")
                st.metric("Top-class softmax score",f"{conf*100:.2f}%")
                sdf=pd.DataFrame([{"Label":k,"Language":LANGS.get(k,k),"Score (%)":v*100} for k,v in scores.items()]).sort_values("Score (%)",ascending=False)
                st.bar_chart(sdf.set_index("Label")["Score (%)"]); st.dataframe(sdf,hide_index=True,use_container_width=True)
                st.caption(f"Model loaded from: {path}")
            except Exception as e: st.error(f"Could not run {chosen}: {e}")

with bench:
    st.subheader("Upload your evaluation sheet")
    st.write("CSV or Excel with required columns text and label. Extra columns such as case_id, language, source_type and has_number are retained.")
    up=st.file_uploader("Evaluation file",type=["csv","xlsx","xls"],key="evaluation_file")
    chosen_models=st.multiselect("Models to evaluate",list(MODELS),default=list(MODELS))
    if up:
        try:
            df=read_eval(up)
            st.success(f"Loaded {len(df):,} labeled rows")
            st.dataframe(df.head(8),use_container_width=True)
            if len(df)>50000: st.warning("Large file: CPU evaluation can take several minutes.")
            if st.button("Run benchmark for selected models",type="primary"):
                results=[]; failures=[]
                for n in chosen_models:
                    st.markdown(f"#### {n}")
                    try:
                        with st.spinner(f"Evaluating {n}..."): result=evaluate(n,df)
                        results.append(result); st.success(f"{n} complete")
                    except Exception as e: failures.append({"Model":n,"Error":str(e)}); st.error(f"{n} failed: {e}")
                if results:
                    st.session_state["v44_results"]=results
                    summary=pd.DataFrame([{"Model":x["name"],"Accuracy":x["accuracy"],"Macro precision":x["macro_precision"],"Macro recall":x["macro_recall"],"Macro F1":x["macro_f1"],"Weighted F1":x["weighted_f1"],"P50 batch ms/row":x["p50_ms_row"],"P90 batch ms/row":x["p90_ms_row"],"Rows":len(df)} for x in results]).sort_values("Macro F1",ascending=False)
                    st.subheader("Leaderboard")
                    st.dataframe(summary.style.format({c:"{:.4f}" for c in ["Accuracy","Macro precision","Macro recall","Macro F1","Weighted F1"]}),hide_index=True,use_container_width=True)
                    st.subheader("Metric comparison")
                    st.bar_chart(summary.set_index("Model")[["Accuracy","Macro F1","Weighted F1"]])
                    win=summary.iloc[0]
                    st.success(f"Best by macro-F1 on this evaluation sheet: {win['Model']} ({win['Macro F1']:.4f})")
                    st.caption("This is the best model on this file only. Check per-class recall, confusion matrices and production latency before selecting a winner.")
                    for x in results:
                        with st.expander(f"{x['name']} detailed metrics and confusion matrix"):
                            report=pd.DataFrame(x["report"]).T.reset_index().rename(columns={"index":"Class"})
                            st.dataframe(report,use_container_width=True)
                            fig,ax=plt.subplots(figsize=(10,8))
                            ConfusionMatrixDisplay(x["cm"],display_labels=x["labels"]).plot(ax=ax,xticks_rotation=45,colorbar=False,values_format="d")
                            ax.set_title(f"{x['name']} confusion matrix"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    combined=df.copy()
                    for x in results:
                        key=x["name"].lower().replace(" ","_")
                        combined[f"prediction_{key}"]=x["preds"]; combined[f"confidence_{key}"]=x["confidence"]
                    st.download_button("Download combined row-level predictions",combined.to_csv(index=False).encode("utf-8-sig"),"version_4_4_benchmark_predictions.csv","text/csv")
                if failures: st.dataframe(pd.DataFrame(failures),hide_index=True,use_container_width=True)
        except Exception as e: st.error(f"Could not read evaluation sheet: {e}")
    else:
        st.info("Your realworld_eval_set.csv format is supported: it has text and label columns.")

st.divider()
st.caption("Benchmark latency is batch inference time divided by batch size, a throughput estimate rather than true per-request p50/p90. For production latency, run a separate single-request benchmark.")
