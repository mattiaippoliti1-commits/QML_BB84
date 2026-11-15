import numpy as np
from src.channels import canale_gilbert_elliott

# Spiegazione: il canale ha due stati, G e B. In G la probabilità di passare a B è a, 
# in B la probabilità di passare a G è b. La frazione di tempo in B è e_B = a/(a+b). 
# Vogliamo che e_B sia circa 0.10, e che la durata media delle raffiche in B sia circa 40 slot 
# (quindi b = 1/40 = 0.025). La durata media degli intervalli in G sarà circa 1/a.

rng = np.random.default_rng(42)
q_t, stati, _ = canale_gilbert_elliott(4000, 0.03, 0.10, 0.005, 40, rng)

# Lancio della moneta truccata slot per slot: errore con probabilità q_t
errori = rng.random(4000) < q_t

# Riga 'stato' e riga 'errori' una sotto l'altra, 100 slot per riga
for inizio in range(600, 1300, 100):
    s = stati[inizio:inizio + 100]
    e = errori[inizio:inizio + 100]
    print(f"{inizio:4d} stato  " + "".join("B" if x else "." for x in s))
    print(f"     errori " + "".join("x" if x else "." for x in e))
    print()

print(f"errori dentro le raffiche: {errori[stati].sum()} su {stati.sum()} slot in B")
print(f"errori fuori dalle raffiche: {errori[~stati].sum()} su {(~stati).sum()} slot in G")
print(f"QBER della sessione: {errori.mean():.4f}   (q medio = 0.03)")