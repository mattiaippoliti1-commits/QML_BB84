"""
Test dei modelli di canale (Fase 2, passo 2.3).

Come in test_bb84.py, i controlli statistici confrontano una stima con il
valore teorico usando una tolleranza di 5 deviazioni standard: un test
corretto fallisce per sfortuna circa una volta su 3 milioni.
"""
import numpy as np
import pytest

from src.bb84 import genera_sessioni, genera_sessioni_canale, qber_atteso
# from src.channels import (canale_gilbert_elliott, canale_iid, genera_stati,
#                           parametri_ge)

from src.channels import (canale_ge_variabile, canale_gilbert_elliott, canale_iid,
                          canale_iid_variabile, genera_stati, parametri_ge)

# Parametri di riferimento: quelli dell'esercizio del passo 2.1
Q, PI_B, E_G, L_B = 0.03, 0.10, 0.005, 40
N, N_PUB = 4000, 1000


# ---------------------------------------------------------------------------
# 1. Calibrazione: le tre formule
# ---------------------------------------------------------------------------

def test_parametri_esercizio():
    """I valori calcolati a mano nel passo 2.1."""
    a, b, e_B = parametri_ge(Q, PI_B, E_G, L_B)
    assert b == pytest.approx(0.025)
    assert a == pytest.approx(0.025 * 0.1 / 0.9)
    assert e_B == pytest.approx(0.255)


def test_bilancio_dei_flussi():
    """All'equilibrio il flusso G->B uguaglia il flusso B->G."""
    a, b, e_B = parametri_ge(Q, PI_B, E_G, L_B)
    assert (1 - PI_B) * a == pytest.approx(PI_B * b)


def test_qber_medio_ricostruito():
    """La media pesata degli errori dei due stati restituisce q_medio."""
    a, b, e_B = parametri_ge(Q, PI_B, E_G, L_B)
    assert (1 - PI_B) * E_G + PI_B * e_B == pytest.approx(Q)


# ---------------------------------------------------------------------------
# 2. La catena di Markov degli stati
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def catena_lunga():
    """Una sola catena molto lunga, generata una volta e riusata dai test."""
    a, b, _ = parametri_ge(Q, PI_B, E_G, L_B)
    rng = np.random.default_rng(12345)
    return genera_stati(500_000, a, b, PI_B, rng), a, b


def test_frazione_in_B(catena_lunga):
    """La frazione di tempo in B tende a pi_B."""
    stati, a, b = catena_lunga
    lam = 1 - a - b
    M = stati.size
    # Gli stati sono correlati: la varianza della media è più grande del caso
    # indipendente di un fattore (1 + lam) / (1 - lam), qui circa 70.
    sigma = np.sqrt(PI_B * (1 - PI_B) / M * (1 + lam) / (1 - lam))
    assert abs(stati.mean() - PI_B) < 5 * sigma


def lunghezze_raffiche(stati):
    """Durata (in slot) di ogni tratto consecutivo in B."""
    # Aggiungo un False ai due estremi, così ogni raffica ha un inizio e una fine
    bordi = np.diff(np.concatenate(([False], stati, [False])).astype(int))
    inizi = np.flatnonzero(bordi == 1)
    fini = np.flatnonzero(bordi == -1)
    return fini - inizi


def test_durata_media_raffiche(catena_lunga):
    """Le raffiche durano in media L_B = 1/b slot (distribuzione geometrica)."""
    stati, a, b = catena_lunga
    durate = lunghezze_raffiche(stati)
    sigma = np.sqrt(1 - b) / b / np.sqrt(durate.size)  # dev. std della media
    assert abs(durate.mean() - L_B) < 5 * sigma


# Tolleranze = circa 5 volte la dispersione della stima misurata su 30 seed
# diversi: più k è grande, più la stima è rumorosa.
@pytest.mark.parametrize("k, tolleranza", [(1, 0.005), (10, 0.035), (40, 0.065)])
def test_memoria_lambda(catena_lunga, k, tolleranza):
    """La correlazione tra lo stato in t e in t+k vale lambda^k."""
    stati, a, b = catena_lunga
    lam = 1 - a - b
    s = stati.astype(float)
    corr = np.corrcoef(s[:-k], s[k:])[0, 1]
    assert corr == pytest.approx(lam**k, abs=tolleranza)


def test_caso_limite_senza_memoria():
    """Con a = b = 0.5 (lambda = 0) gli stati sono indipendenti: lancio di moneta."""
    a, b, _ = parametri_ge(q_medio=0.03, pi_B=0.5, e_G=0.005, L_B=2)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    stati = genera_stati(200_000, a, b, 0.5, np.random.default_rng(1)).astype(float)
    corr = np.corrcoef(stati[:-1], stati[1:])[0, 1]
    assert abs(corr) < 5 / np.sqrt(stati.size)


def test_stati_riproducibili():
    """Stesso seed, stessa sequenza di stati."""
    a, b, _ = parametri_ge(Q, PI_B, E_G, L_B)
    s1 = genera_stati(N, a, b, PI_B, np.random.default_rng(7))
    s2 = genera_stati(N, a, b, PI_B, np.random.default_rng(7))
    assert np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# 3. I canali completi
# ---------------------------------------------------------------------------

def test_canale_iid():
    q_t, stati, _ = canale_iid(N, Q, np.random.default_rng(0))
    assert q_t.shape == (N,) and np.all(q_t == Q)
    assert stati is None


def test_canale_ge_valori_di_q():
    """q_t vale e_B negli slot in B ed e_G negli slot in G, nient'altro."""
    _, _, e_B = parametri_ge(Q, PI_B, E_G, L_B)
    q_t, stati, _ = canale_gilbert_elliott(N, Q, PI_B, E_G, L_B, np.random.default_rng(0))
    assert np.all(q_t[stati] == e_B)
    assert np.all(q_t[~stati] == E_G)


def test_ge_degenere_diventa_iid():
    """Se e_G = q_medio allora anche e_B = q_medio: il canale GE è i.i.d."""
    q_t, _, _ = canale_gilbert_elliott(N, Q, PI_B, Q, L_B, np.random.default_rng(0))
    assert np.allclose(q_t, Q)


# ---------------------------------------------------------------------------
# 4. Integrazione con il simulatore BB84
# ---------------------------------------------------------------------------

def canale_ge(N, rng):
    return canale_gilbert_elliott(N, Q, PI_B, E_G, L_B, rng)


def canale_i(N, rng):
    return canale_iid(N, Q, rng)


@pytest.fixture(scope="module")
def sessioni_ge_attacco():
    return genera_sessioni_canale(500, N, 0.10, canale_ge, N_PUB, seed=2026)


def test_qber_medio_ge_con_eve(sessioni_ge_attacco):
    """Anche con il canale GE il QBER medio segue q_medio + f(1-2 q_medio)/4."""
    qber = np.array([pub.qber for pub, _ in sessioni_ge_attacco])
    sigma = qber.std(ddof=1) / np.sqrt(qber.size)
    assert abs(qber.mean() - qber_atteso(0.10, Q)) < 5 * sigma


def test_ogni_sessione_ha_le_sue_raffiche(sessioni_ge_attacco):
    """La trappola del passo D: il canale va estratto di nuovo per ogni sessione."""
    stati = [nas.stati_canale for _, nas in sessioni_ge_attacco[:20]]
    assert all(s is not None and s.shape == (N,) for s in stati)
    # nessuna coppia di sessioni ha la stessa sequenza di stati
    assert len({s.tobytes() for s in stati}) == len(stati)


def test_sovradispersione_ge():
    """Senza Eve, il QBER del canale GE oscilla molto più di quello binomiale."""
    sessioni = genera_sessioni_canale(500, N, 0.0, canale_ge, N_PUB, seed=99)
    qber = np.array([pub.qber for pub, _ in sessioni])
    std_binomiale = np.sqrt(Q * (1 - Q) / N_PUB)
    assert qber.std() > 1.5 * std_binomiale


def test_iid_compatibile_con_fase_1():
    """Con il canale i.i.d. la nuova funzione riproduce genera_sessioni."""
    vecchie = genera_sessioni(5, N, 0.1, Q, N_PUB, seed=7)
    nuove = genera_sessioni_canale(5, N, 0.1, canale_i, N_PUB, seed=7)
    for (p1, _), (p2, n2) in zip(vecchie, nuove):
        assert np.array_equal(p1.pos, p2.pos)
        assert np.array_equal(p1.errori, p2.errori)
        assert n2.stati_canale is None

# ---------------------------------------------------------------------------
# 5. Rumore variabile da sessione a sessione
# ---------------------------------------------------------------------------
Q_MIN, Q_MAX, RAPPORTO_EG = 0.01, 0.05, 1 / 6


def canale_iid_var(N, rng):
    return canale_iid_variabile(N, Q_MIN, Q_MAX, rng)


def canale_ge_var(N, rng):
    return canale_ge_variabile(N, Q_MIN, Q_MAX, PI_B, RAPPORTO_EG, L_B, rng)


def test_q_variabile_uniforme_nell_intervallo():
    """q_medio cade in [Q_MIN, Q_MAX] e la sua media è il centro dell'intervallo."""
    rng = np.random.default_rng(5)
    q = np.array([canale_iid_variabile(10, Q_MIN, Q_MAX, rng)[2] for _ in range(20_000)])
    assert q.min() >= Q_MIN and q.max() <= Q_MAX
    sigma = (Q_MAX - Q_MIN) / np.sqrt(12) / np.sqrt(q.size)  # dev. std della media
    assert abs(q.mean() - (Q_MIN + Q_MAX) / 2) < 5 * sigma


def test_ge_variabile_mantiene_la_forma():
    """e_G è sempre q_medio/6, e e_B resta una probabilità sensata (<= 0.5)."""
    rng = np.random.default_rng(6)
    for _ in range(50):
        q_t, stati, q = canale_ge_variabile(N, Q_MIN, Q_MAX, PI_B, RAPPORTO_EG, L_B, rng)
        assert np.allclose(q_t[~stati], q * RAPPORTO_EG)
        assert np.all(q_t <= 0.5)


def test_ge_variabile_coincide_con_il_fisso_a_3_percento():
    """Con q_medio = 3% l'opzione B dà e_G = 0.5%: lo stesso canale di prima."""
    assert 0.03 * RAPPORTO_EG == pytest.approx(E_G)


@pytest.mark.parametrize("canale", [canale_iid_var, canale_ge_var])
def test_stesso_rumore_sotto_H0_e_H1(canale):
    """Con lo stesso seed, H0 e H1 ricevono ESATTAMENTE gli stessi q_medio:
    il rumore non può tradire la presenza di Eve."""
    h0 = genera_sessioni_canale(50, N, 0.0, canale, N_PUB, seed=31)
    h1 = genera_sessioni_canale(50, N, 0.10, canale, N_PUB, seed=31)
    assert [n.q_medio for _, n in h0] == [n.q_medio for _, n in h1]