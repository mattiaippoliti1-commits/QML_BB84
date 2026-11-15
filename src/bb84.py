"""
Simulatore BB84 (Fase 1 della roadmap).

Riunisce i passi 1-6:
  Alice -> [Eve, intercept-resend su una frazione f] -> canale rumoroso -> Bob
  -> sifting -> campione pubblico di n_pub bit.

Convenzioni
  basi:  0 = Z (|0>, |1>),  1 = X (|+>, |->)
  f:     probabilità che Eve intercetti un qubit
  q:     probabilità che il canale inverta il bit di un qubit (dopo Eve).
         Può essere un numero (rumore i.i.d.) o un array lungo N con un
         valore per slot (rumore non i.i.d., es. Gilbert-Elliott in Fase 2).

Separazione fondamentale
  Pubblico -> ciò che Alice e Bob si dicono: è l'UNICO input dei rivelatori.
  Nascosto -> ground truth della simulazione: serve solo per analisi e test,
              non deve MAI diventare una feature.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pubblico:
    pos: np.ndarray      # slot temporali dei bit confrontati (ordinati)
    basi: np.ndarray     # base usata per ciascuno (0 = Z, 1 = X)
    errori: np.ndarray   # True se Alice e Bob hanno bit diversi
    n_sifted: int        # quanti bit sono sopravvissuti al sifting

    @property
    def qber(self) -> float:
        return float(self.errori.mean())


@dataclass
class Nascosto:
    f: float                   # parametro dell'attacco usato
    intercettato: np.ndarray   # quali slot Eve ha intercettato
    sift_pos: np.ndarray       # tutti gli slot tenuti dopo il sifting
    sift_errori: np.ndarray    # errori su TUTTI i bit tenuti

    @property
    def qber_vero(self) -> float:
        return float(self.sift_errori.mean())


def simula_sessione(N, f, q, n_pub, rng):
    """Simula una sessione con N impulsi. `rng` è un np.random.Generator."""
    # --- Alice ---
    alice_bits  = rng.integers(0, 2, N)
    alice_bases = rng.integers(0, 2, N)

    # --- Eve: intercetta ogni qubit con probabilità f, misura in base casuale ---
    intercettato = rng.random(N) < f
    eve_bases = rng.integers(0, 2, N)
    eve_bits  = np.where(eve_bases == alice_bases, alice_bits, rng.integers(0, 2, N))

    # qubit che prosegue verso Bob: quello rispedito da Eve oppure quello di Alice
    base_inviata = np.where(intercettato, eve_bases, alice_bases)
    bit_inviato  = np.where(intercettato, eve_bits,  alice_bits)

    # --- Canale: inverte il bit con probabilità q (la base non cambia) ---
    bit_ricevuto = bit_inviato ^ (rng.random(N) < q)

    # --- Bob: stessa base del qubit ricevuto -> bit certo, altrimenti casuale ---
    bob_bases = rng.integers(0, 2, N)
    bob_bits  = np.where(bob_bases == base_inviata, bit_ricevuto, rng.integers(0, 2, N))

    # --- Sifting ---
    sift_pos = np.flatnonzero(alice_bases == bob_bases)
    sift_errori = alice_bits[sift_pos] != bob_bits[sift_pos]
    if n_pub > sift_pos.size:
        raise ValueError(f"n_pub={n_pub} ma solo {sift_pos.size} bit dopo il sifting")

    # --- Campione pubblico ---
    pub_pos = np.sort(rng.choice(sift_pos, size=n_pub, replace=False))
    pubblico = Pubblico(
        pos=pub_pos,
        basi=alice_bases[pub_pos],
        errori=alice_bits[pub_pos] != bob_bits[pub_pos],
        n_sifted=int(sift_pos.size),
    )
    nascosto = Nascosto(f=f, intercettato=intercettato,
                        sift_pos=sift_pos, sift_errori=sift_errori)
    return pubblico, nascosto


def genera_sessioni(n_sessioni, N, f, q, n_pub, seed):
    """
    Genera n_sessioni sessioni indipendenti e riproducibili.
    Ogni sessione riceve un proprio generatore, derivato da `seed` con
    SeedSequence.spawn: i flussi casuali sono indipendenti tra loro e la
    sessione i è identica qualunque sia n_sessioni.
    """
    figli = np.random.SeedSequence(seed).spawn(n_sessioni)
    return [simula_sessione(N, f, q, n_pub, np.random.default_rng(s)) for s in figli]


def qber_atteso(f, q):
    """QBER teorico: (1-f)·q + f·[½·q + ½·½] = q + f(1-2q)/4."""
    return q + f * (1 - 2 * q) / 4

# genera_sessioni_canale: come genera_sessioni, ma il rumore viene da un modello di canale.

def genera_sessioni_canale(n_sessioni, N, f, canale, n_pub, seed):
    """
    Come genera_sessioni, ma il rumore viene da un modello di canale.
    `canale` è una funzione canale(N, rng) -> (q_t, stati): viene chiamata
    UNA VOLTA PER SESSIONE, con il generatore di quella sessione, così ogni
    sessione ha le sue raffiche.
    """
    figli = np.random.SeedSequence(seed).spawn(n_sessioni)
    sessioni = []
    for s in figli:
        rng = np.random.default_rng(s)
        q_t, stati = canale(N, rng)                              # 1. rumore di questa sessione
        pubblico, nascosto = simula_sessione(N, f, q_t, n_pub, rng)  # 2. protocollo
        nascosto.stati_canale = stati                            # 3. ground truth del canale
        sessioni.append((pubblico, nascosto))
    return sessioni




# Nascosto: contiene il ground truth della simulazione, non deve mai essere usato come feature.
@dataclass
class Nascosto:
    f: float
    intercettato: np.ndarray
    sift_pos: np.ndarray
    sift_errori: np.ndarray
    stati_canale: np.ndarray | None = None   # NUOVO: stati G/B (None per il canale i.i.d.)