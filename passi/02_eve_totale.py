import numpy as np

# Simulazione del protocollo BB84 con intercettazione totale di Eve

rng = np.random.default_rng(0)
N = 100_000          # tanti qubit, così le frequenze si avvicinano alle probabilità

# --- Alice ---
alice_bits  = rng.integers(0, 2, N)
alice_bases = rng.integers(0, 2, N)

# --- Eve: misura ogni qubit in una base casuale ---
eve_bases = rng.integers(0, 2, N)
eve_same  = (eve_bases == alice_bases)
# stessa base di Alice -> legge il bit giusto; base diversa -> esito casuale
eve_bits  = np.where(eve_same, alice_bits, rng.integers(0, 2, N))
# Eve rispedisce lo stato che ha trovato: d'ora in poi il qubit
# è codificato nella base di EVE con il bit di EVE

# --- Bob ---
bob_bases = rng.integers(0, 2, N)
bob_same  = (bob_bases == eve_bases)      # attenzione: confronto con la base di EVE
bob_bits  = np.where(bob_same, eve_bits, rng.integers(0, 2, N))

# --- Sifting: Alice e Bob confrontano le LORO basi (di Eve non sanno nulla) ---
sift = (alice_bases == bob_bases)
errori = (alice_bits[sift] != bob_bits[sift])

print("bit tenuti:", sift.sum(), "su", N)
print("QBER:", errori.mean())

# Verifica dell'albero: separiamo i bit tenuti in base a cosa ha fatto Eve
eve_ok = eve_same[sift]
print("QBER quando Eve ha indovinato la base:", errori[eve_ok].mean())
print("QBER quando Eve ha sbagliato la base: ", errori[~eve_ok].mean())