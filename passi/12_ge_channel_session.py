import numpy as np
from src.bb84 import genera_sessioni, genera_sessioni_canale, qber_atteso
from src.channels import canale_iid, canale_gilbert_elliott

N, N_PUB, Q = 4000, 1000, 0.03

# I due canali, con i parametri fissati: restano solo (N, rng), la forma
# che si aspetta genera_sessioni_canale
def iid(N, rng):
    return canale_iid(N, Q, rng)

def ge(N, rng):
    return canale_gilbert_elliott(N, Q, 0.10, 0.005, 40, rng)

print(f"{'canale':6s} {'f':>5s} {'QBER medio':>11s} {'atteso':>8s} {'dev. std':>9s}")
for nome, canale in [("iid", iid), ("GE", ge)]:
    for f in [0.0, 0.10]:
        sessioni = genera_sessioni_canale(2000, N, f, canale, N_PUB, seed=2026)
        qber = np.array([pub.qber for pub, nas in sessioni])
        print(f"{nome:6s} {f:5.2f} {qber.mean():11.4f} {qber_atteso(f, Q):8.4f} {qber.std():9.4f}")
print(f"dev. std binomiale teorica (f=0): {np.sqrt(Q*(1-Q)/N_PUB):.4f}")

# Compatibilità: con il canale i.i.d. la nuova funzione dà gli STESSI
# risultati di genera_sessioni (canale_iid non consuma numeri casuali)
a = genera_sessioni(5, N, 0.1, Q, N_PUB, seed=7)
b = genera_sessioni_canale(5, N, 0.1, iid, N_PUB, seed=7)
print("identici:", all(np.array_equal(x[0].errori, y[0].errori) for x, y in zip(a, b)))