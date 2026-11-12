# Fase 1 — Simulatore BB84 (completata il 2026-11-12)

Codice scritto passo per passo da Mattia nel repository `bb84_qml` (il pacchetto zip iniziale `bb84sim` è stato abbandonato a favore di questa versione).

## Struttura
- `passi/01…08_*.py`: percorso didattico (BB84 base, Eve totale, Eve parziale f, rumore i.i.d., rumore a raffiche Gilbert-Elliott, campione pubblico, Qiskit).
- `src/bb84.py`: `simula_sessione(N, f, q, n_pub, rng)` → (`Pubblico`, `Nascosto`); `genera_sessioni(...)` con `SeedSequence.spawn`; `qber_atteso(f, q)`.
- `tests/test_bb84.py`: 23 test pytest, tutti superati.
- Ambiente: venv `.qml` con Python 3.13.2, qiskit 2.6.2, qiskit-aer 0.18.2.

## Scelte di modellazione
- Notazione della roadmap: f = frazione di qubit intercettati, q = probabilità di inversione del bit nel canale.
- Rumore applicato dopo Eve, indipendente dalla base (equivale a una porta Y con probabilità q). QBER atteso = q + f(1−2q)/4.
- `q` può essere scalare (i.i.d.) o un array per slot (non i.i.d.): il canale Gilbert-Elliott della Fase 2 si collega senza modificare `bb84.py`.
- Il rivelatore vede solo `Pubblico` (posizioni, basi, errori di n_pub bit, n_sifted); `Nascosto` è ground truth e non deve mai diventare una feature.
- Sessione tipica: N = 4000 impulsi → ~2000 sifted → n_pub = 1000.

## Validazione
- Numpy: QBER coincide con la teoria entro 5σ su una griglia di valori di q e f; sifting ≈ ½; riproducibilità con seed.
- Qiskit (passo 8, AerSimulator): probabilità a singolo qubit corrette; QBER di sessione entro 5σ dalla teoria; negli slot deterministici gli esiti di Bob coincidono al 100% con la previsione.

## Lezioni metodologiche emerse
- Stesso seed tra le esecuzioni → fluttuazioni correlate: train e test dovranno usare seed principali diversi.
- Separazione ≈ (f/4) / √(q(1−q)/n): con f = 0,02 e n = 1000 è circa 1σ (quasi indistinguibile); gli scenari utili stanno intorno a 1–3σ.
- Canale i.i.d. + Eve uniforme: il QBER è statistica sufficiente, quindi la soglia è già ottima (Neyman-Pearson). Il possibile vantaggio sta nel canale a raffiche.