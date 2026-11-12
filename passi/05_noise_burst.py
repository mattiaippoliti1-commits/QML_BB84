import numpy as np

def stati_canale(N, p_gb, p_bg, rng):
    """Catena di Markov a due stati: 0 = Good, 1 = Bad. Un passo per slot."""
    stato = np.zeros(N, dtype=int)
    s = 0                                   # si parte da Good
    for i in range(N):
        u = rng.random()
        if s == 0 and u < p_gb:             # Good -> Bad
            s = 1
        elif s == 1 and u < p_bg:           # Bad -> Good
            s = 0
        stato[i] = s
    return stato

def simula(eta, q_per_slot, N, rng):
    """Come il passo 4, ma q può essere diverso per ogni slot.
    Ritorna (posizioni dei bit tenuti, array degli errori su quei bit)."""
    alice_bits  = rng.integers(0, 2, N)
    alice_bases = rng.integers(0, 2, N)
    intercettato = rng.random(N) < eta
    eve_bases = rng.integers(0, 2, N)
    eve_bits  = np.where(eve_bases == alice_bases, alice_bits, rng.integers(0, 2, N))
    base_inviata = np.where(intercettato, eve_bases, alice_bases)
    bit_inviato  = np.where(intercettato, eve_bits,  alice_bits)
    disturbato   = rng.random(N) < q_per_slot      # confronto slot per slot
    bit_ricevuto = bit_inviato ^ disturbato
    bob_bases = rng.integers(0, 2, N)
    bob_bits  = np.where(bob_bases == base_inviata, bit_ricevuto, rng.integers(0, 2, N))
    sift = (alice_bases == bob_bases)
    posizioni = np.flatnonzero(sift)               # slot dei bit tenuti
    errori = alice_bits[sift] != bob_bits[sift]
    return posizioni, errori

def qber_a_blocchi(posizioni, errori, N, blocco):
    """QBER calcolato separatamente su blocchi consecutivi di `blocco` slot."""
    return [errori[(posizioni >= a) & (posizioni < a + blocco)].mean()
            for a in range(0, N, blocco)]

# --- Parametri ---
N = 20_000
q_good, q_bad = 0.01, 0.15
p_bg = 1 / 50                                       # raffica lunga in media 50 slot
pi_bad = (0.03 - q_good) / (q_bad - q_good)         # frazione di tempo in Bad per avere media 3%
p_gb = pi_bad * p_bg / (1 - pi_bad)

rng = np.random.default_rng(1)
stato = stati_canale(N, p_gb, p_bg, rng)
q_ge  = np.where(stato == 1, q_bad, q_good)         # q diverso per ogni slot
q_iid = np.full(N, 0.03)                            # stesso q = 3% per tutti

print(f"frazione di tempo in Bad: {stato.mean():.3f} (atteso {pi_bad:.3f})")
for nome, q in [("i.i.d.         ", q_iid), ("Gilbert-Elliott", q_ge)]:
    pos, err = simula(0.0, q, N, rng)
    blocchi = qber_a_blocchi(pos, err, N, 1000)
    print(f"{nome}  QBER totale {err.mean():.3f}   per blocchi:",
          " ".join(f"{b:.3f}" for b in blocchi))