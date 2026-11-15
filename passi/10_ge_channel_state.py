import numpy as np
from src.channels import parametri_ge, genera_stati

# Passo A: calibrazione
a, b, e_B = parametri_ge(0.03, 0.10, 0.005, 40)
print(f"a = {a:.5f}   b = {b:.3f}   e_B = {e_B:.3f}")

# Passo B: una sessione da 4000 slot
rng = np.random.default_rng(42)
stati = genera_stati(4000, a, b, 0.10, rng)

# Disegno: '.' = slot in G, 'B' = slot in B, 100 slot per riga (prime 1500)
for inizio in range(0, 1500, 100):
    riga = stati[inizio:inizio + 100]
    print(f"{inizio:4d} " + "".join("B" if s else "." for s in riga))

print(f"frazione di tempo in B: {stati.mean():.3f}   (attesa 0.10)")
inizi = np.sum(~stati[:-1] & stati[1:])          # slot dove si passa da G a B
print(f"raffiche iniziate: {inizi}   (attese circa 10)")

# Media su molte sessioni
frazioni = [genera_stati(4000, a, b, 0.10, rng).mean() for _ in range(500)]
print(f"frazione in B, media su 500 sessioni: {np.mean(frazioni):.3f}")
print(f"  ... ma da una sessione all'altra varia tra {min(frazioni):.3f} e {max(frazioni):.3f}")