import streamlit as st
import joblib, numpy as np, pandas as pd, xgboost as xgb, re
from scipy import sparse

from supabase import create_client, Client

# My Supabase project credentials

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_to_supabase(sequence, drug, resistance, known, rising):
    supabase.table("predictions").insert({
        "input_sequence": sequence,
        "drug": drug,
        "resistance_label": resistance,
        "known_mutations": known,
        "rising_mutations": rising
    }).execute()

# ─────────────────────────────── Load artefacts ───────────────────────────────
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

boosters = []
for i in range(6):                         # boosters 0-5 : 3TC … TDF
    bst = xgb.Booster()
    bst.load_model(f"xgb_booster_label_{i}.json")
    boosters.append(bst)

def boosters_predict(X_csr):
    dmat = xgb.DMatrix(X_csr)
    preds = [np.argmax(bst.predict(dmat), axis=1) for bst in boosters]
    return np.vstack(preds).T            # (n_samples , 6)

# ─────────────────────────────── Constants ────────────────────────────────────
drug_labels = ["3TC", "ABC", "D4T", "AZT", "DDI", "TDF"]
full_names  = {
    "3TC":"Lamivudine","ABC":"Abacavir","D4T":"Stavudine",
    "AZT":"Zidovudine","DDI":"Didanosine","TDF":"Tenofovir"
}
res_labels  = [
    "Susceptible","Potential Low Resistance","Low Resistance",
    "Intermediate Resistance","High Resistance"
]

KNOWN_NRTI_DRMS = {
    "M184V","M184I","K65R","K65N","D67N","D67G","K70R","K70E","K70G","K70N",
    "L74V","L74I","A62V","V75T","F77L","Y115F","Q151M","T69D","T69N","T69I",
    "M41L","L210W","T215Y","T215F","K219Q","K219E"
}

TOP_UNKNOWN_MUTATIONS = ["K122E", "R211K", "V118I", "T200A", "D177E", "K103N", 
               "I135T", "Q207E", "D123E", "F214L", "G196E", "V60I", 
               "Y181C", "K20R", "L228H"]




CROSS_EFFECTS_DETAILED = {
    "M184V":{"effect":"↑ AZT & TDF susceptibility; 3TC resistance",
             "clinical":"Keep AZT/TDF despite 3TC failure."},
    "M184I":{"effect":"Same as M184V (weaker)","clinical":"Hypersensitizes to AZT."},
    "K65R" :{"effect":"↓ TDF, ABC, 3TC susceptibility",
             "clinical":"Avoid TDF/ABC if K65R present."},
    "M41L":{"effect":"TAM → AZT/D4T resistance","clinical":"Part of TAM-1 cluster."},
    "D67N":{"effect":"TAM → AZT/D4T resistance","clinical":"Adds when with others."},
    "K70R":{"effect":"TAM → AZT/D4T resistance","clinical":"Common early TAM."},
    "L210W":{"effect":"TAM → AZT/D4T resistance","clinical":"Enhances TAM pattern."},
    "T215Y":{"effect":"Major TAM, high AZT resistance","clinical":"Key TAM-1 mutation."},
    "T215F":{"effect":"TAM similar to T215Y","clinical":"High AZT resistance."},
    "K219Q":{"effect":"Minor TAM","clinical":"Adds to AZT resistance."},
    "K219E":{"effect":"Minor TAM","clinical":"Adds to AZT resistance."},
    "A62V":{"effect":"Part of Q151M complex","clinical":"Broad NRTI resistance."},
    "T69D":{"effect":"Indicates multi-NRTI pattern","clinical":"Little alone."},
    "L74V":{"effect":"↓ ABC & DDI susceptibility","clinical":"Compromises ABC/ddI."}
}

WT = ("PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKL"
      "VDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL"
      "PQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKE"
      "PPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAE"
      "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKI"
      "ATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRET"
      "KLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQL"
      "IKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL")[:240]

# ─────────────────────────────── Helpers ───────────────────────────────────────
def list_mutations(seq: str):
    seq = re.sub(r"[^A-Za-z]", "", seq.upper())
    seq = (seq + WT[len(seq):])[:240]
    return [f"{wt}{i}{aa}" for i,(wt,aa) in enumerate(zip(WT, seq),1) if wt!=aa]

def kmers(seq: str, k=5):
    return " ".join(seq[i:i+k] for i in range(len(seq)-k+1))

def mutation_notes(muts):
    if not muts: return "None"
    lines=[]
    for m in muts:
        if m in CROSS_EFFECTS_DETAILED:
            info=CROSS_EFFECTS_DETAILED[m]
            lines.append(f"**{m}** — {info['effect']}  \n_{info['clinical']}_")
        else:
            lines.append(f"**{m}** — clinical impact unknown")
    return "\n".join(lines)

# ─────────────────────────────── Streamlit UI ──────────────────────────────────
st.title("🧬 HIV-1 NRTI Resistance Predictor")
st.markdown(
"""Paste an RT amino-acid sequence (100-240 aa).  
The model predicts resistance to **3TC, ABC, D4T, AZT, DDI, TDF** and lists mutations."""
)
# Add example sequences dictionary (or list)
examples = {
    "Example 1 (All drugs)": "MLWQTKVTVLDVGDAYFSVPLDLEGKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDLEGKWRKLVDFRELNKRTQDFWEVQLGVKHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL",
    "Example 2 (3TC)": "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL"
}

st.sidebar.title("Load example sequence")
selected_example = st.sidebar.selectbox("Choose an example sequence", options=["-- None --"] + list(examples.keys()))

# Default sequence to empty string or chosen example
default_seq = examples[selected_example] if selected_example != "-- None --" else ""

# Use default_seq in the text area below
seq = st.text_area("RT sequence:", height=180, value=default_seq, placeholder="PQITLWQRPLVTIKIGG...")
filter_choice = st.multiselect("Show drugs:", drug_labels, default=drug_labels)

if st.button("Predict"):
    # --- validate and predict ---
    if len(re.sub(r"[^A-Za-z]", "", seq)) < 100:
        st.error("Sequence must contain at least 100 amino acids.")
        st.stop()

    km = kmers(re.sub(r"[^A-Za-z]", "", seq.upper()))
    X = tfidf_vectorizer.transform([km])
    preds = boosters_predict(X)[0]

    # --- mutation analysis ---
    muts = list_mutations(seq)
    known = [m for m in muts if m in KNOWN_NRTI_DRMS]
    rising = [m for m in muts if m in TOP_UNKNOWN_MUTATIONS]

    # --- resistance table ---
    rows = [
        {"Drug": drug, "Resistance": res_labels[pred]}
        for drug, pred in zip(drug_labels, preds)
        if drug in filter_choice
    ]
    # Log each drug’s prediction to Supabase
    for drug, pred in zip(drug_labels, preds):
        if drug in filter_choice:
            log_to_supabase(
                sequence=seq,
                drug=drug,
                resistance=res_labels[pred],
                known=known,
                rising=rising
            )
    st.subheader("Resistance predictions")
    st.dataframe(pd.DataFrame(rows).set_index("Drug"))

    # --- known DRMs ---
    st.subheader("Known DRMs detected")
    st.markdown(mutation_notes(known))

    # --- rising mutations ---
    st.subheader("Rising mutations")
    if rising:
        st.markdown(
            f"**Detected rising mutations in sequence:** {', '.join(rising)}"
        )
    else:
        st.markdown("No rising mutations from the top-15 list detected in this sequence.")

