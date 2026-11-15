"""
Passo 9 — Quanto è affidabile il QBER?

Simula molte sessioni BB84: in metà (a caso) c'è Eve, nelle altre no.
Poi guarda la distribuzione del QBER stimato nei due gruppi e chiede:
  - quali sono le sessioni con QBER più basso e più alto, e c'era Eve?
  - quanto si sovrappongono i due gruppi?
  - una soglia sul QBER quanti attacchi scopre, se si accetta l'1% di falsi allarmi?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bb84 import qber_atteso, simula_sessione  # noqa: E402

# --- Parametri dell'esperimento ---
N_SESSIONI = 1_000_000
N, N_PUB = 4000, 1000        # impulsi per sessione, bit confrontati
Q = 0.03                     # rumore del canale (i.i.d.)
F = 0.05                     # frazione intercettata quando Eve c'è
P_EVE = 0.5                  # probabilità che in una sessione ci sia Eve
SEED = 2026

# Ogni sessione ha il suo generatore indipendente; un generatore a parte
# decide in quali sessioni c'è Eve.
figli = np.random.SeedSequence(SEED).spawn(N_SESSIONI + 1)
eve_presente = np.random.default_rng(figli[0]).random(N_SESSIONI) < P_EVE

qber = np.empty(N_SESSIONI)
for i in range(N_SESSIONI):
    f = F if eve_presente[i] else 0.0
    pub, _ = simula_sessione(N, f, Q, N_PUB, np.random.default_rng(figli[i + 1]))
    qber[i] = pub.qber          # SOLO informazione pubblica

senza, con = qber[~eve_presente], qber[eve_presente]
etichetta = lambda e: "CON Eve" if e else "senza Eve"

# --- 1. Medie e fluttuazioni ---
print(f"{N_SESSIONI} sessioni: {(~eve_presente).sum()} senza Eve, {eve_presente.sum()} con Eve (f={F})\n")
for nome, v, f in [("senza Eve", senza, 0.0), ("con Eve  ", con, F)]:
    m = qber_atteso(f, Q)
    print(f"{nome}: QBER medio {v.mean():.4f} (teoria {m:.4f})   "
          f"dev.std {v.std():.4f} (teoria {np.sqrt(m*(1-m)/N_PUB):.4f})")

# --- 2. Caso migliore e peggiore ---
ordine = np.argsort(qber)
print("\n5 sessioni con QBER più BASSO:")
for i in ordine[:5]:
    print(f"   sessione {i:5d}   QBER {qber[i]:.4f}   {etichetta(eve_presente[i])}")
print("5 sessioni con QBER più ALTO:")
for i in ordine[-5:][::-1]:
    print(f"   sessione {i:5d}   QBER {qber[i]:.4f}   {etichetta(eve_presente[i])}")

# --- 3. Sovrapposizione: i casi che ingannano ---
print(f"\nQBER più alto SENZA Eve: {senza.max():.4f}")
print(f"QBER più basso CON Eve:  {con.min():.4f}")
nella_zona = ((qber >= con.min()) & (qber <= senza.max())).mean()
print(f"sessioni nella zona di sovrapposizione: {nella_zona:.1%}")
print(f"sessioni con Eve sotto la media delle sessioni senza Eve: {(con < senza.mean()).mean():.1%}")

# --- 4. Una soglia con l'1% di falsi allarmi ---
soglia = np.quantile(senza, 0.99)   # l'1% delle sessioni senza Eve la supera
fpr = (senza > soglia).mean()
tpr = (con > soglia).mean()
print(f"\nSoglia al 99° percentile delle sessioni senza Eve: {soglia:.4f}")
print(f"falsi allarmi (FPR): {fpr:.1%}   attacchi scoperti (TPR): {tpr:.1%}")