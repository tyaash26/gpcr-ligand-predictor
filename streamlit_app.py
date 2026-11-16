# GPCR ligand activity predictor — Streamlit app.
# Training-free "labels as features" method (reproduction of Hansson et al.,
# Nat. Commun. 2025; data Zenodo 12532113, CC-BY 4.0). Research/education only.
import numpy as np, streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title='GPCR Ligand Predictor', page_icon='🧬')

@st.cache_resource
def load():
    d = np.load('predictor_data.npz', allow_pickle=True)
    fps = np.unpackbits(d['fps'], axis=1).astype(np.float32)   # (N, 2048)
    return fps, fps.sum(1), d['labels'], list(d['receptors']), list(d['classes'])

FPS, FPNORM, LAB, REC, CLASSES = load()
CODE2CLS = {i+1: c for i, c in enumerate(CLASSES)}
ACTIVE = set(range(1, 6))   # 1-5 active classes; 6 = No Effect

st.title('🧬 GPCR ligand activity predictor')
st.caption("Predicts a compound's likely bioactivity class on GPCRs from its nearest "
           "chemical neighbours' labels — a training-free reproduction of the "
           "labels-as-features method of Hansson et al., Nat. Commun. 2025. "
           "Data: Zenodo 12532113 (CC-BY 4.0). Reference set subsampled to ~40k compounds. "
           "Research / education only.")

smi = st.text_input('Enter a SMILES', 'CC(=O)Oc1ccccc1C(=O)O')
topk = st.slider('Neighbours to vote (k)', 10, 100, 50)

if st.button('Predict', type='primary'):
    m = Chem.MolFromSmiles((smi or '').strip())
    if m is None:
        st.error('Could not parse that SMILES — please check it.')
    else:
        arr = np.zeros((2048,), dtype=np.uint8)
        ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048), arr)
        q = arr.astype(np.float32)
        inter = FPS @ q
        tani = inter / (FPNORM + q.sum() - inter + 1e-9)
        order = np.argsort(-tani)[:topk]
        rows = []
        for j, R in enumerate(REC):
            ln = LAB[order, j]; lab = ln[ln > 0]
            if len(lab) < 3: continue
            vals, counts = np.unique(lab, return_counts=True); k = counts.argmax()
            code, conf = vals[k], counts[k]/len(lab)
            if code in ACTIVE:
                rows.append({'receptor': R, 'predicted class': CODE2CLS[code],
                             'confidence': round(float(conf),2), 'neighbours': int(len(lab))})
        if rows:
            rows = sorted(rows, key=lambda r: -r['confidence'])
            st.success(f'{len(rows)} receptor(s) predicted active · top neighbour Tanimoto {tani[order[0]]:.2f}')
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info('No confident active prediction (few labelled neighbours for this chemotype).')
