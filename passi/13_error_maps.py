"""
Figure del passo 2.4: come appaiono gli errori nel campione pubblico.

Figura 1 (mappa degli errori): una sessione per riga, i.i.d. e GE allo stesso
QBER, senza e con Eve. Ogni trattino verticale è un errore nel campione pubblico.
Figura 2 (istogrammi): distribuzione del QBER pubblico su molte sessioni.

Si lancia dalla cartella principale:  python3 -m passi.13_error_maps
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.bb84 import genera_sessioni_canale, qber_atteso
from src.channels import canale_gilbert_elliott, canale_iid

N, N_PUB, Q, F = 4000, 1000, 0.03, 0.10
CARTELLA = Path("results/figures")
CARTELLA.mkdir(parents=True, exist_ok=True)


def iid(N, rng):
    return canale_iid(N, Q, rng)


def ge(N, rng):
    return canale_gilbert_elliott(N, Q, 0.10, 0.005, 40, rng)


def sessione_tipica(canale, f, seed):
    """Tra 200 sessioni sceglie quella con QBER pubblico più vicino al valore atteso.

    Così le righe della figura hanno lo stesso QBER e l'occhio può confrontare
    solo la DISPOSIZIONE degli errori, non la loro quantità.
    """
    sessioni = genera_sessioni_canale(200, N, f, canale, N_PUB, seed)
    qber = np.array([pub.qber for pub, _ in sessioni])
    i = np.argmin(np.abs(qber - qber_atteso(f, Q)))
    return sessioni[i]


def intervalli_in_B(stati):
    """(inizio, fine) di ogni raffica: servono solo per colorare lo sfondo."""
    bordi = np.diff(np.concatenate(([False], stati, [False])).astype(int))
    return zip(np.flatnonzero(bordi == 1), np.flatnonzero(bordi == -1))


# --------------------------------------------------------------------------
# Figura 1: mappa degli errori
# --------------------------------------------------------------------------
righe = [
    ("i.i.d., senza Eve", iid, 0.0),
    ("GE, senza Eve", ge, 0.0),
    (f"i.i.d., Eve f = {F}", iid, F),
    (f"GE, Eve f = {F}", ge, F),
]

fig, assi = plt.subplots(len(righe), 1, figsize=(11, 5.5), sharex=True)
for ax, (titolo, canale, f) in zip(assi, righe):
    pub, nas = sessione_tipica(canale, f, seed=11)

    # Sfondo: raffiche vere del canale. È ground truth (Nascosto): la usiamo
    # SOLO per spiegare la figura, il rivelatore non la vedrà mai.
    if nas.stati_canale is not None:
        for inizio, fine in intervalli_in_B(nas.stati_canale):
            ax.axvspan(inizio, fine, color="tab:orange", alpha=0.25, lw=0)

    # Errori nel campione pubblico: un trattino nella posizione (slot) dell'errore
    ax.eventplot(pub.pos[pub.errori], colors="black", lineoffsets=0.5,
                 linelengths=0.8, linewidths=1)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel(titolo, rotation=0, ha="right", va="center", fontsize=9)
    ax.text(1.005, 0.5, f"QBER {pub.qber:.3f}\n{pub.errori.sum()} errori",
            transform=ax.transAxes, va="center", fontsize=8)

assi[-1].set_xlabel("slot (posizione temporale dell'impulso)")
assi[0].set_title("Errori nel campione pubblico (n = 1000), stesso QBER medio; "
                  "arancione = raffiche del canale GE (nascoste al rivelatore)",
                  fontsize=10)
fig.tight_layout()
fig.savefig(CARTELLA / "error_maps.png", dpi=150)
print("salvata", CARTELLA / "error_maps.png")

# --------------------------------------------------------------------------
# Figura 2: istogrammi del QBER su molte sessioni
# --------------------------------------------------------------------------
fig, assi = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
# Il QBER può valere solo k/1000 (k errori su 1000 bit): i bordi delle barre
# vanno messi a metà tra due valori possibili, altrimenti alcune barre
# contengono 1 valore e altre 3 e l'istogramma viene "a pettine".
bins = (np.arange(0, 102, 2) - 0.5) / N_PUB
for ax, (nome, canale) in zip(assi, [("canale i.i.d.", iid), ("canale GE", ge)]):
    for f, colore, etichetta in [(0.0, "tab:blue", "senza Eve (H0)"),
                                 (F, "tab:red", f"Eve f = {F} (H1)")]:
        sessioni = genera_sessioni_canale(2000, N, f, canale, N_PUB, seed=2026)
        qber = np.array([pub.qber for pub, _ in sessioni])
        ax.hist(qber, bins=bins, color=colore, alpha=0.5, label=etichetta)
        ax.axvline(qber_atteso(f, Q), color=colore, ls="--", lw=1)
    ax.set_title(nome)
    ax.set_xlabel("QBER del campione pubblico")
assi[0].set_ylabel("numero di sessioni (su 2000)")
assi[0].legend()
fig.suptitle("Stesso QBER medio, stessa Eve: con le raffiche le due distribuzioni "
             "si sovrappongono molto di più", fontsize=10)
fig.tight_layout()
fig.savefig(CARTELLA / "qber_histograms.png", dpi=150)
print("salvata", CARTELLA / "qber_histograms.png")