import numpy as np

def sessione(f, q, N, n_pub, rng):
    """Una sessione BB84 completa. Ritorna SOLO ciò che è pubblico:
    posizioni, basi ed errori degli n_pub bit confrontati."""
    alice_bits  = rng.integers(0, 2, N)
    alice_bases = rng.integers(0, 2, N)
    intercettato = rng.random(N) < f
    eve_bases = rng.integers(0, 2, N)
    eve_bits  = np.where(eve_bases == alice_bases, alice_bits, rng.integers(0, 2, N))
    base_inviata = np.where(intercettato, eve_bases, alice_bases)
    bit_inviato  = np.where(intercettato, eve_bits,  alice_bits)
    bit_ricevuto = bit_inviato ^ (rng.random(N) < q)
    bob_bases = rng.integers(0, 2, N)
    bob_bits  = np.where(bob_bases == base_inviata, bit_ricevuto, rng.integers(0, 2, N))

    # Sifting: posizioni (slot) dei bit con la stessa base
    sift_pos = np.flatnonzero(alice_bases == bob_bases)

    # Campione pubblico: n_pub posizioni scelte a caso tra quelle tenute (NUOVO)
    pub_pos  = np.sort(rng.choice(sift_pos, size=n_pub, replace=False))
    pub_basi = alice_bases[pub_pos]
    pub_err  = alice_bits[pub_pos] != bob_bits[pub_pos]
    return pub_pos, pub_basi, pub_err


rng = np.random.default_rng(0)
f, q, N, n_pub = 0.1, 0.03, 4000, 1000     # 4000 impulsi -> ~2000 tenuti -> 1000 pubblici

# Una sessione
pos, basi, err = sessione(f, q, N, n_pub, rng)
print(f"bit pubblici: {len(err)}   QBER: {err.mean():.4f}   "
      f"QBER base Z: {err[basi==0].mean():.4f}   QBER base X: {err[basi==1].mean():.4f}")

# 2000 sessioni: quanto fluttua il QBER stimato?
atteso = q + f*(1-2*q)/4
qber = np.array([sessione(f, q, N, n_pub, rng)[2].mean() for _ in range(2000)])
print(f"2000 sessioni: media {qber.mean():.4f} (teoria {atteso:.4f})   "
      f"dev.std {qber.std():.4f} (teoria {np.sqrt(atteso*(1-atteso)/n_pub):.4f})")