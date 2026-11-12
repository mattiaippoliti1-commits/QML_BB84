import numpy as np

def simula(eta, q, N=100_000, seed=0):
    """Sessione BB84: Eve intercetta con prob. eta, poi il canale
    inverte il bit di ogni qubit con prob. q (i.i.d.)."""
    rng = np.random.default_rng(seed)

    # --- Alice ---
    alice_bits  = rng.integers(0, 2, N)
    alice_bases = rng.integers(0, 2, N)

    # --- Eve ---
    intercettato = rng.random(N) < eta
    eve_bases = rng.integers(0, 2, N)
    eve_bits  = np.where(eve_bases == alice_bases, alice_bits, rng.integers(0, 2, N))

    base_inviata = np.where(intercettato, eve_bases, alice_bases)
    bit_inviato  = np.where(intercettato, eve_bits,  alice_bits)

    # --- Canale rumoroso (NUOVO) ---
    disturbato = rng.random(N) < q              # stesso trucco di eta: True con prob. q
    bit_ricevuto = bit_inviato ^ disturbato     # ^ = XOR: inverte il bit dove disturbato è True
    # la base non cambia: |0>,|1> restano in Z, |+>,|-> restano in X

    # --- Bob ---
    bob_bases = rng.integers(0, 2, N)
    bob_bits  = np.where(bob_bases == base_inviata, bit_ricevuto, rng.integers(0, 2, N))

    # --- Sifting ---
    sift = (alice_bases == bob_bases)
    return (alice_bits[sift] != bob_bits[sift]).mean()


def teoria(eta, q):
    return q + eta * (1 - 2*q) / 4


print(" eta     q    QBER sim   teoria")
for eta, q in [(0, 0), (0, 0.025), (0, 0.05), (0.1, 0), (0.1, 0.05), (1, 0.05)]:
    print(f"{eta:4.1f} {q:6.3f}    {simula(eta, q):.4f}   {teoria(eta, q):.4f}")