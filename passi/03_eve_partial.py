import numpy as np

def simula(eta, N=100_000, seed=0):
    """Una sessione BB84 in cui Eve intercetta ogni qubit con probabilità eta.
    Ritorna il QBER sui bit tenuti dopo il sifting."""
    rng = np.random.default_rng(seed)

    # --- Alice ---
    alice_bits  = rng.integers(0, 2, N)
    alice_bases = rng.integers(0, 2, N)

    # --- Eve: decide QUALI qubit intercettare ---
    intercettato = rng.random(N) < eta     # True con probabilità eta, per ogni qubit

    # Eve "misura" tutti i qubit (semplifica il codice);
    # useremo il suo risultato solo dove intercettato è True
    eve_bases = rng.integers(0, 2, N)
    eve_same  = (eve_bases == alice_bases)
    eve_bits  = np.where(eve_same, alice_bits, rng.integers(0, 2, N))

    # --- Qubit che arriva a Bob ---
    # intercettato -> lo stato rispedito da Eve (base e bit di Eve)
    # non intercettato -> lo stato originale di Alice
    base_inviata = np.where(intercettato, eve_bases, alice_bases)
    bit_inviato  = np.where(intercettato, eve_bits,  alice_bits)

    # --- Bob ---
    bob_bases = rng.integers(0, 2, N)
    bob_same  = (bob_bases == base_inviata)   # conta la base del qubit che RICEVE
    bob_bits  = np.where(bob_same, bit_inviato, rng.integers(0, 2, N))

    # --- Sifting (Alice e Bob confrontano le LORO basi) ---
    sift = (alice_bases == bob_bases)
    return (alice_bits[sift] != bob_bits[sift]).mean()

# Simulazione per diversi valori di eta, ossia probabilità che Eve intercetti un singolo qubit
for eta in [0.0, 0.1, 0.5, 1.0]:
    print(f"eta = {eta:.1f}   QBER simulato = {simula(eta):.4f}   teoria = {eta/4:.4f}")

# Risultati attesi:
# Con eta = 0.0, Eve non intercetta nulla, quindi QBER = 0
# Con eta = 0.1, Eve intercetta il 10% dei qubit, quindi QBER = 0.025
# Con eta = 0.5, Eve intercetta il 50% dei qubit, quindi QBER = 0.125
# Con eta = 1.0, Eve intercetta tutto, quindi QBER = 0.25