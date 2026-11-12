import numpy as np


# Simulazione del protocollo BB84 senza rumore e senza intercettazioni

rng = np.random.default_rng(0)      # seed fisso: stesso risultato a ogni esecuzione
N = 20                              # numero di qubit inviati

alice_bits  = rng.integers(0, 2, N) # bit casuali di Alice
alice_bases = rng.integers(0, 2, N) # 0 = base Z, 1 = base X
bob_bases   = rng.integers(0, 2, N) # basi casuali di Bob

same_basis = (alice_bases == bob_bases)   # array di True/False, uno per qubit
coin = rng.integers(0, 2, N)              # esito casuale, usato solo se la base è diversa
bob_bits = np.where(same_basis, alice_bits, coin)
#          ^ per ogni qubit: se same_basis è True prendi alice_bits, altrimenti coin

# Sifting: Alice e Bob si dicono pubblicamente le basi e tengono solo i bit con la stessa base
key_alice = alice_bits[same_basis]
key_bob   = bob_bits[same_basis]

print("basi Alice:", alice_bases)
print("basi Bob:  ", bob_bases)
print("bit tenuti:", same_basis.sum(), "su", N)
print("probabilità: ", same_basis.sum() / N * 100, "%")
print("errori nella chiave:", (key_alice != key_bob).sum())

# Calcolo dei bit scartati
scartati_giusti = (alice_bits[~same_basis] == bob_bits[~same_basis]).sum()
print("bit scartati:", (~same_basis).sum(), "- di cui giusti per caso:", scartati_giusti)