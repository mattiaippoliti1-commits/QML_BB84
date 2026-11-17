"""
Difficoltà degli scenari per la soglia sul QBER.

Per ogni scenario della griglia (configs/scenarios.yaml):
  1. simula sessioni SENZA Eve e mette la soglia al quantile (1 - FPR) del QBER;
  2. simula sessioni CON Eve e misura il TPR (frazione sopra la soglia);
  3. classifica lo scenario: impossibile / difficile / facile.

È la soglia "ideale" (calibrata su molte sessioni H0): il meglio che la soglia
sul QBER possa fare. Serve a scegliere gli scenari, non è il risultato finale.

Uscite:
  results/scenario_difficulty.csv      tabella completa
  results/figures/threshold_tpr.png    TPR della soglia in funzione di f

Si lancia dalla cartella principale:  python3 -m scripts.scenario_difficulty
"""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.bb84 import genera_sessioni_canale
from src.scenari import carica_config, costruisci_canale, griglia

CARTELLA = Path("results")
(CARTELLA / "figures").mkdir(parents=True, exist_ok=True)

cfg = carica_config()
val = cfg["valutazione"]


def qber_sessioni(scenario, f, n_sessioni, seed):
    """QBER pubblico di n_sessioni sessioni dello scenario, con frazione di attacco f."""
    canale = costruisci_canale(scenario.rumore, cfg)
    sessioni = genera_sessioni_canale(n_sessioni, scenario.N, f, canale, scenario.n, seed)
    return np.array([pub.qber for pub, _ in sessioni])


def classe(tpr):
    if tpr <= val["tpr_impossibile"]:
        return "impossibile"
    if tpr >= val["tpr_facile"]:
        return "facile"
    return "difficile"


righe = []
cache_h0 = {}   # le sessioni H0 non dipendono da f: si simulano una volta per (rumore, n)
inizio = time.time()

for scenario in griglia(cfg):
    i_rumore = cfg["rumore"].index(scenario.rumore)
    i_f = cfg["f"].index(scenario.f)

    # Seed distinti e riproducibili: [seed base, rumore, n, H0/H1, f].
    # H0 e H1 non condividono mai lo stesso flusso casuale.
    chiave_h0 = (scenario.rumore, scenario.n)
    if chiave_h0 not in cache_h0:
        cache_h0[chiave_h0] = qber_sessioni(
            scenario, 0.0, val["sessioni_h0"], [val["seed"], i_rumore, scenario.n, 0])
    q0 = cache_h0[chiave_h0]
    q1 = qber_sessioni(scenario, scenario.f, val["sessioni_h1"],
                       [val["seed"], i_rumore, scenario.n, 1, i_f])

    riga = {"rumore": scenario.rumore, "n": scenario.n, "f": scenario.f}
    for fpr in val["fpr"]:
        soglia = np.quantile(q0, 1 - fpr)
        # Allarme se il QBER SUPERA la soglia. Il QBER è discreto (k/n), quindi
        # l'FPR reale può essere un po' sotto quello nominale: lo registriamo.
        riga[f"fpr_reale@{fpr}"] = float(np.mean(q0 > soglia))
        riga[f"tpr@{fpr}"] = float(np.mean(q1 > soglia))
    riga["classe"] = classe(riga[f"tpr@{val['fpr_classificazione']}"])
    righe.append(riga)
    print(f"{scenario.nome:28s} TPR@5% = {riga['tpr@0.05']:.3f}  {riga['classe']:11s}"
          f"  ({time.time() - inizio:5.0f} s)")

# --------------------------------------------------------------------------
# Tabella
# --------------------------------------------------------------------------
with open(CARTELLA / "scenario_difficulty.csv", "w", newline="") as fp:
    scrittore = csv.DictWriter(fp, fieldnames=list(righe[0]))
    scrittore.writeheader()
    scrittore.writerows(righe)
print("salvata", CARTELLA / "scenario_difficulty.csv")

# --------------------------------------------------------------------------
# Figura: un pannello per configurazione di rumore, una curva per n
# --------------------------------------------------------------------------
fpr_c = val["fpr_classificazione"]
fig, assi = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
for ax, rumore in zip(assi, cfg["rumore"]):
    ax.axhspan(0, val["tpr_impossibile"], color="tab:red", alpha=0.08)
    ax.axhspan(val["tpr_facile"], 1.02, color="tab:green", alpha=0.08)
    ax.axhline(fpr_c, color="gray", ls=":", lw=1)   # TPR di un rivelatore a caso
    for n in cfg["n"]:
        sel = [r for r in righe if r["rumore"] == rumore and r["n"] == n]
        ax.plot([r["f"] for r in sel], [r[f"tpr@{fpr_c}"] for r in sel], "o-", label=f"n = {n}")
    ax.set_title(rumore.replace("_", ", "))
    ax.set_xlabel("f (frazione intercettata)")
    ax.set_xticks(cfg["f"])
    ax.set_ylim(0, 1.02)
assi[0].set_ylabel(f"TPR della soglia (FPR = {fpr_c:.0%})")
assi[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Difficoltà degli scenari per la soglia sul QBER "
             "(rosso: impossibile, verde: facile, puntinato: rivelatore a caso)", fontsize=10)
fig.tight_layout()
fig.savefig(CARTELLA / "figures" / "threshold_tpr.png", dpi=150)
print("salvata", CARTELLA / "figures" / "threshold_tpr.png")