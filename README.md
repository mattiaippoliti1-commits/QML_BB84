# BB84 + QML: detecting intercept-resend attacks

Simulation of the BB84 quantum key distribution protocol with an
intercept-resend eavesdropper, used to compare a classical QBER-threshold
detector with a Quantum Machine Learning classifier.

**Research question.** Can a QML model detect an attack on a quantum key
distribution session better than a threshold on the quantum bit error rate
(QBER), at the same false-alarm rate?

## Setup

Requires Python 3.13.

```bash
python3 -m venv .qml
source .qml/bin/activate
pip install -r requirements.txt
```

## Usage

All commands are run from the repository root.

```bash
python3 -m pytest                      # run the whole test suite
python3 -m passi.12_ge_channel_session # run a step-by-step script (no .py)
```

## Repository structure

```
src/
  bb84.py       BB84 simulator: Alice, Eve (intercept-resend), channel, Bob,
                sifting, public sample
  channels.py   channel models: i.i.d. and Gilbert-Elliott (burst noise)
tests/          pytest test suite
passi/          step-by-step scripts that build and check the simulator
```

## Model

**Session.** Alice sends `N` qubits. Eve intercepts each one with
probability `f` and measures it in a random basis. The channel then flips
each bit with probability `q_t`. After sifting, Alice and Bob compare a public
sample of `n_pub` bits.

**Public vs hidden data.** Each simulated session returns two objects:

- `Pubblico`: what Alice and Bob reveal (positions, bases, errors of the
  public sample). It is the **only** input a detector may use.
- `Nascosto`: simulation ground truth (intercepted qubits, all sifted errors,
  channel states). Used for tests and analysis only, never as a feature.

**Channel models.** Both produce a per-slot flip probability `q_t`:

- *i.i.d.*: `q_t = q` for every slot.
- *Gilbert-Elliott*: a two-state Markov chain (good state G, bad state B)
  producing bursts of errors. It is parametrised by the mean QBER `q̄`, the
  fraction of time in B `π_B`, the error rate in G `e_G` and the mean burst
  length `L_B`, so that it can be calibrated to the **same mean QBER** as the
  i.i.d. channel: the two channels differ only in how errors are arranged.

**Expected QBER** with attack fraction `f`: `q̄ + f(1 − 2q̄)/4`.

## Reproducibility

Every session gets its own random generator, derived from a master seed with
`numpy.random.SeedSequence.spawn`. The same seed always gives the same
sessions; training and test data must use different master seeds.