"""
Test del simulatore BB84 (Fase 1). Eseguire dalla radice del repository con:
    python -m pytest -v

I test statistici accettano scarti fino a 5 deviazioni standard: con seed fissi
il risultato è deterministico, e un fallimento indica un errore vero.
"""
import numpy as np
import pytest

from bb84 import genera_sessioni, qber_atteso, simula_sessione


def tolleranza(p, n):
    """5 deviazioni standard di una frequenza p stimata su n prove."""
    return 5 * np.sqrt(p * (1 - p) / n)


def test_senza_eve_senza_rumore_zero_errori():
    pub, nas = simula_sessione(N=10_000, f=0.0, q=0.0, n_pub=1000,
                               rng=np.random.default_rng(1))
    assert nas.qber_vero == 0.0
    assert pub.qber == 0.0


def test_sifting_tiene_circa_meta():
    N = 200_000
    pub, _ = simula_sessione(N, 0.0, 0.0, 1000, np.random.default_rng(2))
    assert abs(pub.n_sifted / N - 0.5) < tolleranza(0.5, N)


@pytest.mark.parametrize("f", [0.0, 0.02, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("q", [0.0, 0.03, 0.10])
def test_qber_coincide_con_la_teoria(f, q):
    _, nas = simula_sessione(200_000, f, q, 1000, np.random.default_rng(3))
    p = qber_atteso(f, q)
    assert abs(nas.qber_vero - p) < tolleranza(p, nas.sift_errori.size) + 1e-12


def test_eve_totale_da_25_percento():
    _, nas = simula_sessione(200_000, 1.0, 0.0, 1000, np.random.default_rng(4))
    assert abs(nas.qber_vero - 0.25) < tolleranza(0.25, nas.sift_errori.size)


def test_campione_pubblico_ben_formato():
    pub, nas = simula_sessione(4000, 0.1, 0.03, 1000, np.random.default_rng(5))
    assert pub.pos.size == 1000
    assert np.all(np.diff(pub.pos) > 0)               # ordinato, senza ripetizioni
    assert np.all(np.isin(pub.pos, nas.sift_pos))     # solo bit sopravvissuti al sifting


def test_n_pub_troppo_grande_da_errore():
    with pytest.raises(ValueError):
        simula_sessione(100, 0.0, 0.0, 1000, np.random.default_rng(6))


def test_q_per_slot_accettato():
    # rumore solo nella prima metà degli slot: gli errori devono cadere lì
    N = 20_000
    q = np.where(np.arange(N) < N // 2, 0.2, 0.0)
    _, nas = simula_sessione(N, 0.0, q, 1000, np.random.default_rng(7))
    assert np.all(nas.sift_pos[nas.sift_errori] < N // 2)


def test_riproducibilita():
    a = genera_sessioni(5, 4000, 0.1, 0.03, 1000, seed=42)
    b = genera_sessioni(8, 4000, 0.1, 0.03, 1000, seed=42)
    for (pa, _), (pb, _) in zip(a, b[:5]):
        assert np.array_equal(pa.pos, pb.pos)
        assert np.array_equal(pa.errori, pb.errori)


def test_sessioni_diverse_sono_diverse():
    sessioni = genera_sessioni(2, 4000, 0.1, 0.03, 1000, seed=42)
    assert not np.array_equal(sessioni[0][0].errori, sessioni[1][0].errori)