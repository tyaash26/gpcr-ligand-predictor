# GPCR ligand activity predictor — Streamlit app (v2: reports its own reliability).
# Training-free "labels as features" method (reproduction of Hansson et al.,
# Nat. Commun. 2025; data Zenodo 12532113, CC-BY 4.0). Research/education only.
import numpy as np, streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title='GPCR Ligand Predictor', page_icon='🧬')

# Rule-of-thumb Tanimoto thresholds (Morgan r=2, 2048 bits). NOT calibrated yet:
# the natural next step is to measure held-out accuracy as a function of similarity.
CLOSE, WEAK = 0.60, 0.40

@st.cache_resource
def load():
    d = np.load('predictor_data.npz', allow_pickle=True)
    fps = np.unpackbits(d['fps'], axis=1).astype(np.float32)   # (N, 2048)
    return fps, fps.sum(1), d['labels'], list(d['receptors']), list(d['classes'])

FPS, FPNORM, LAB, REC, CLASSES = load()
CODE2CLS = {i + 1: c for i, c in enumerate(CLASSES)}
ACTIVE = set(range(1, 6))   # codes 1-5 = active classes; 6 = No Effect

st.title('🧬 GPCR ligand activity predictor')
st.caption("Predicts a compound's likely bioactivity class on GPCRs from the known labels of its "
           "nearest chemical neighbours. Reference set subsampled to ~40k compounds. "
           "Research / education only.")

smi = st.text_input('Enter a SMILES', 'CC(=O)Oc1ccccc1C(=O)O')
topk = st.slider('Neighbours to vote (k)', 10, 100, 50)

if st.button('Predict', type='primary'):
    m = Chem.MolFromSmiles((smi or '').strip())
    if m is None:
        st.error('Could not parse that SMILES — please check it.')
    else:
        arr = np.zeros((2048,), dtype=np.uint8)
        ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), arr)
        q = arr.astype(np.float32)
        inter = FPS @ q
        tani = inter / (FPNORM + q.sum() - inter + 1e-9)
        order = np.argsort(-tani)[:topk]
        sims = tani[order]
        top = float(sims[0])

        # --- how well does the reference set cover this query? ---
        c1, c2, c3 = st.columns(3)
        c1.metric('Nearest-neighbour similarity', f'{top:.2f}')
        c2.metric(f'Median similarity (top {topk})', f'{float(np.median(sims)):.2f}')
        c3.metric('Reference compounds', f'{len(FPS):,}')
        if top >= CLOSE:
            st.success('A close analogue exists in the reference set — predictions are close to a lookup.')
        elif top >= WEAK:
            st.warning('Only moderately similar compounds exist in the reference set — treat predictions as hints.')
        else:
            st.error('⚠ No similar compound in the reference set (nearest-neighbour similarity < '
                     f'{WEAK:.2f}). The query is outside the well-covered chemical space: any receptor '
                     'listed below is an extrapolation, not a reliable prediction.')

        rows = []
        for j, R in enumerate(REC):
            ln = LAB[order, j]
            has = ln > 0
            lab, s = ln[has], sims[has]
            if len(lab) < 3:
                continue
            vals, counts = np.unique(lab, return_counts=True)
            k = counts.argmax(); code = vals[k]
            if code not in ACTIVE:
                continue
            support = float(s[lab == code].max())          # closest neighbour backing the call
            rel = 'close analogue' if support >= CLOSE else ('moderate' if support >= WEAK else 'weak')
            rows.append({'receptor': R, 'predicted class': CODE2CLS[code],
                         'neighbour agreement': round(float(counts[k] / len(lab)), 2),
                         'voters': int(len(lab)),
                         'closest supporting neighbour': round(support, 2),
                         'reliability': rel})
        if rows:
            order_rel = {'close analogue': 0, 'moderate': 1, 'weak': 2}
            rows.sort(key=lambda r: (order_rel[r['reliability']], -r['closest supporting neighbour'],
                                     -r['neighbour agreement']))
            st.dataframe(rows, hide_index=True)
        else:
            st.info('No receptor reached a confident active call (too few labelled neighbours for this chemotype).')

with st.expander('How to read this'):
    st.markdown(
        "- **Neighbour agreement** is the fraction of labelled neighbours that voted for the winning class. "
        "It is *not* a probability of being correct, and with few voters it is brittle (e.g. 4 of 4 = 1.0).\n"
        "- **Closest supporting neighbour** is the Tanimoto similarity of the most similar compound backing the call. "
        "Close analogue (≥ 0.60) behaves like a database lookup; weak (< 0.40) is extrapolation.\n"
        "- Thresholds are rule-of-thumb and not yet calibrated against held-out accuracy.\n"
        "- The reference set is small-molecule bioactivity data from ChEMBL; large peptides and proteins are "
        "mostly outside it, so expect low similarity (and weak predictions) for those.")

st.caption("(Method reproduced from Hansson et al., Nat. Commun. 2025 · "
           "data: Zenodo 12532113, CC-BY 4.0)")
