import streamlit as st
import joblib, numpy as np, pandas as pd, xgboost as xgb, re
from scipy import sparse

# ─────────────────────────────── Load artefacts ───────────────────────────────
tfidf_vectorizer = joblib.load("tfidf_vectorizer_fixed.pkl")

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

seq = st.text_area("RT sequence:", height=180, placeholder="PQITLWQRPLVTIKIGG...")
filter_choice = st.multiselect("Show drugs:", drug_labels, default=drug_labels)

if st.button("Predict"):
    if len(re.sub(r"[^A-Za-z]","",seq)) < 100:
        st.error("Sequence must contain at least 100 amino acids.")
        st.stop()

    km = kmers(re.sub(r"[^A-Za-z]","",seq.upper()))
    X  = tfidf_vectorizer.transform([km])
    preds = boosters_predict(X)[0]

    muts = list_mutations(seq)
    known = [m for m in muts if m in KNOWN_NRTI_DRMS]
    unknown = [m for m in muts if m not in KNOWN_NRTI_DRMS]

    rows=[]
    for drug,pred in zip(drug_labels, preds):
        if drug not in filter_choice: continue
        rows.append({
            "Drug": drug,
            "Resistance": res_labels[pred]
        })
    st.subheader("Resistance predictions")
    st.dataframe(pd.DataFrame(rows).set_index("Drug"))

    st.subheader("Known DRMs detected")
    st.markdown(mutation_notes(known))

    st.subheader("Other mutations")
    st.markdown(", ".join(unknown) if unknown else "None")
