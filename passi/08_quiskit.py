"""
Passo 8 — BB84 con circuiti Qiskit veri, per validare il simulatore numpy.

Parte A: un singolo qubit, eseguito molte volte (shots) per vedere le probabilità.
Parte B: una sessione intera, un circuito per slot temporale; confronto con la teoria
         e controllo esatto sugli slot in cui l'esito è deterministico.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

Z, X = 0, 1
sim = AerSimulator(seed_simulator=1234)   # seed fisso: risultati riproducibili


def circuito_bb84(bit, base_alice, base_bob, base_eve=None, rumore=False):
    """
    Un qubit, due bit classici: c[0] = risultato di Eve, c[1] = risultato di Bob.
    base_eve=None significa che Eve non intercetta questo qubit.
    """
    qc = QuantumCircuit(1, 2)

    # --- Alice prepara ---
    if bit == 1:
        qc.x(0)                 # |0> -> |1>
    if base_alice == X:
        qc.h(0)                 # |0> -> |+>,  |1> -> |->
    qc.barrier()

    # --- Eve (se presente): misura nella sua base e rispedisce ---
    if base_eve is not None:
        if base_eve == X:
            qc.h(0)             # ruota la base X sulla base Z...
        qc.measure(0, 0)        # ...misura (lo stato collassa)...
        if base_eve == X:
            qc.h(0)             # ...e ruota indietro: lo stato rispedito è |+> o |->
        qc.barrier()

    # --- Canale: rumore ---
    if rumore:
        qc.y(0)                 # Y inverte il bit sia in base Z sia in base X
        qc.barrier()

    # --- Bob misura nella sua base ---
    if base_bob == X:
        qc.h(0)
    qc.measure(0, 1)
    return qc


def leggi(memoria, k):
    """Qiskit scrive i bit classici da destra a sinistra: c[0] è l'ultimo carattere."""
    return int(memoria[::-1][k])


# =============================================================================
# Parte A — un qubit, 2000 esecuzioni per caso
# =============================================================================
print("PARTE A: frequenze del risultato di Bob (Alice invia sempre il bit 0)\n")
casi = [
    ("base Z, Bob Z, senza Eve",        dict(bit=0, base_alice=Z, base_bob=Z)),
    ("base X, Bob Z, senza Eve",        dict(bit=0, base_alice=X, base_bob=Z)),
    ("base Z, Bob Z, Eve misura in Z",  dict(bit=0, base_alice=Z, base_bob=Z, base_eve=Z)),
    ("base Z, Bob Z, Eve misura in X",  dict(bit=0, base_alice=Z, base_bob=Z, base_eve=X)),
    ("base Z, Bob Z, rumore Y",         dict(bit=0, base_alice=Z, base_bob=Z, rumore=True)),
    ("base X, Bob X, rumore Y",         dict(bit=0, base_alice=X, base_bob=X, rumore=True)),
]
for nome, parametri in casi:
    memoria = sim.run(circuito_bb84(**parametri), shots=2000, memory=True).result().get_memory()
    p1 = np.mean([leggi(m, 1) for m in memoria])
    print(f"  {nome:34s} P(Bob=0) = {1 - p1:.3f}   P(Bob=1) = {p1:.3f}")

print("\nEsempio di circuito (Alice bit 0 in Z, Eve misura in X, rumore, Bob in Z):")
print(circuito_bb84(0, Z, Z, base_eve=X, rumore=True).draw("text"))


# =============================================================================
# Parte B — sessione completa: le scelte classiche con numpy, le misure con Qiskit
# =============================================================================
def sessione_qiskit(N, f, q, rng):
    # tutte le scelte casuali "classiche" (come in src/bb84.py)
    alice_bits   = rng.integers(0, 2, N)
    alice_bases  = rng.integers(0, 2, N)
    intercettato = rng.random(N) < f
    eve_bases    = rng.integers(0, 2, N)
    rumore       = rng.random(N) < q
    bob_bases    = rng.integers(0, 2, N)

    # un circuito per slot; le misure (unica fonte di casualità quantistica) le fa Aer
    circuiti = [circuito_bb84(int(alice_bits[i]), int(alice_bases[i]), int(bob_bases[i]),
                              base_eve=int(eve_bases[i]) if intercettato[i] else None,
                              rumore=bool(rumore[i]))
                for i in range(N)]
    risultato = sim.run(circuiti, shots=1, memory=True).result()
    bob_bits = np.array([leggi(risultato.get_memory(i)[0], 1) for i in range(N)])

    sift = alice_bases == bob_bases
    errori = alice_bits[sift] != bob_bits[sift]

    # Controllo esatto: dove nessuno ha misurato nella base sbagliata, l'esito di Bob
    # è determinato: bit di Alice, invertito se c'è stato rumore.
    det = sift & (~intercettato | (eve_bases == alice_bases))
    atteso_det = alice_bits[det] ^ rumore[det]
    coincidono = np.array_equal(bob_bits[det], atteso_det)
    return errori.mean(), errori.size, det.sum(), coincidono


print("\nPARTE B: sessioni da 2000 slot simulate con circuiti\n")
print("   f     q    QBER Qiskit   teoria    entro 5σ?   slot deterministici   coincidono?")
rng = np.random.default_rng(0)
for f, q in [(0.0, 0.0), (0.0, 0.05), (0.5, 0.0), (1.0, 0.0), (0.5, 0.05)]:
    qber, n, n_det, ok = sessione_qiskit(2000, f, q, rng)
    teoria = q + f * (1 - 2 * q) / 4
    entro = abs(qber - teoria) < 5 * np.sqrt(max(teoria * (1 - teoria), 1e-12) / n) + 1e-12
    print(f"  {f:.1f}  {q:.2f}     {qber:.4f}     {teoria:.4f}      {'sì' if entro else 'NO'}"
          f"          {n_det:5d}                 {'sì' if ok else 'NO'}")