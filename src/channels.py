# src/channels.py
import numpy as np


def parametri_ge(q_medio, pi_B, e_G, L_B):
    """Dai parametri 'intuitivi' ricava quelli della catena di Markov."""
    b = 1 / L_B                                  # fine raffica
    a = b * pi_B / (1 - pi_B)                    # inizio raffica
    e_B = (q_medio - (1 - pi_B) * e_G) / pi_B    # errore durante la raffica
    return a, b, e_B


def genera_stati(N, a, b, pi_B, rng):
    """Sequenza degli stati del canale: True = B (cattivo), False = G (buono)."""
    stati = np.zeros(N, dtype=bool)
    stati[0] = rng.random() < pi_B
    for t in range(1, N):
        if stati[t - 1]:                         # ero in B
            stati[t] = rng.random() < 1 - b      # resto in B con prob. 1-b
        else:                                    # ero in G
            stati[t] = rng.random() < a          # entro in B con prob. a
    return stati

# canale a raffiche: restituisce q_t (prob. di errore per slot) e gli stati.
def canale_gilbert_elliott(N, q_medio, pi_B, e_G, L_B, rng):
    """Canale a raffiche: restituisce q_t (prob. di errore per slot) e gli stati."""
    a, b, e_B = parametri_ge(q_medio, pi_B, e_G, L_B)
    stati = genera_stati(N, a, b, pi_B, rng)
    q_t = np.where(stati, e_B, e_G)
    return q_t, stati, q_medio

def canale_iid(N, q, rng):
    """Canale i.i.d.: stessa probabilità di errore q per tutti gli slot."""
    q_t = np.full(N, q)   # array di N valori tutti uguali a q
    return q_t, None, q   # nessuno stato nascosto


def canale_iid_variabile(N, q_min, q_max, rng):
    """Canale i.i.d. con q estratto uniformemente in [q_min, q_max] per ogni sessione."""
    q = rng.uniform(q_min, q_max)
    return canale_iid(N, q, rng)


def canale_ge_variabile(N, q_min, q_max, pi_B, rapporto_eG, L_B, rng):
    """Canale GE con QBER medio variabile da sessione a sessione.

    La FORMA del canale resta la stessa (pi_B, L_B e il rapporto e_G / q_medio):
    cambia solo il "volume" del rumore. Con q_medio alto si sporcano anche i
    tratti buoni, così un errore fuori dalle raffiche non è automaticamente Eve.
    """
    q = rng.uniform(q_min, q_max)
    return canale_gilbert_elliott(N, q, pi_B, rapporto_eG * q, L_B, rng)