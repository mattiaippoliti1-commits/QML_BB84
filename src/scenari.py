"""Lettura della griglia degli scenari da configs/scenarios.yaml."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from src.channels import (canale_ge_variabile, canale_gilbert_elliott,
                          canale_iid, canale_iid_variabile)

PERCORSO_DEFAULT = Path("configs/scenarios.yaml")


@dataclass(frozen=True)
class Scenario:
    rumore: str   # una delle quattro configurazioni, es. "ge_fisso"
    f: float      # frazione di qubit intercettati
    n: int        # bit del campione pubblico
    N: int        # impulsi per sessione

    @property
    def nome(self) -> str:
        return f"{self.rumore}_f{self.f}_n{self.n}"


# def carica_config(percorso=PERCORSO_DEFAULT):
#     """Legge il file YAML e lo restituisce come dizionario Python."""
#     with open(percorso) as fp:
#         return yaml.safe_load(fp)

def carica_config(percorso=PERCORSO_DEFAULT):
    """Legge il file YAML e lo restituisce come dizionario Python."""
    with open(percorso) as fp:
        cfg = yaml.safe_load(fp)
    if not cfg:
        # safe_load restituisce None se il file è vuoto o contiene solo commenti
        raise ValueError(f"il file di configurazione {percorso} è vuoto")
    return cfg


def griglia(cfg):
    """Tutte le combinazioni (rumore, f, n): il prodotto cartesiano della griglia."""
    return [
        Scenario(rumore=r, f=f, n=n, N=cfg["rapporto_N_su_n"] * n)
        for r, f, n in product(cfg["rumore"], cfg["f"], cfg["n"])
    ]


def costruisci_canale(nome, cfg):
    """Restituisce la funzione canale(N, rng) corrispondente al nome."""
    q, q_min, q_max = cfg["q_fisso"], cfg["q_min"], cfg["q_max"]
    pi_B, L_B = cfg["ge"]["pi_B"], cfg["ge"]["L_B"]
    rapporto_eG = cfg["ge"]["e_G_a_q_fisso"] / q   # = 1/6 con i valori di default

    canali = {
        "iid_fisso": lambda N, rng: canale_iid(N, q, rng),
        "ge_fisso": lambda N, rng: canale_gilbert_elliott(N, q, pi_B, rapporto_eG * q, L_B, rng),
        "iid_variabile": lambda N, rng: canale_iid_variabile(N, q_min, q_max, rng),
        "ge_variabile": lambda N, rng: canale_ge_variabile(N, q_min, q_max, pi_B, rapporto_eG, L_B, rng),
    }
    if nome not in canali:
        raise ValueError(f"configurazione di rumore sconosciuta: {nome!r}")
    return canali[nome]