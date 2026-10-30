# -*- coding: utf-8 -*-

# Quantum_Cairo


"""
QUANTUM CAIRO V1 — Loto 7/39

Inspiracija:
Hannah Cairo:
A Counterexample to the Mizohata-Takeuchi Conjecture.

Ovo je adaptacija:
- zakrivljena geometrija;
- sume podskupova iste velicine;
- koherentni Fourierovi doprinosi;
- zajednicka amplituda cele kombinacije.

Kod ne reprodukuje matematicki dokaz kontraprimera.

Qiskit:
- sest kubita predstavlja 64 talasne komponente;
- istorija odredjuje ulazne kompleksne amplitude;
- nauceno kvantno kolo menja njihove module i faze;
- eksplicitno fazno kolo proverava zavrsnu interferenciju.

Sedam brojeva predstavlja se njihovom zajednickom
geometrijskom koordinatom. Ne simulira se sistem od 42 kubita.

Za kombinaciju S:
    amplitude(S) = sum_r a_r(history) * character_r(S)

Raspodela:
    P(S | history) = |amplitude(S)|^2 / Z(history)

Normalizacija obuhvata sve kombinacije sedam od 39.
Racuna se preko kompleksnih elementarnih simetricnih polinoma.

NEXT maksimum:
deterministicki pregled svih kombinacija u blokovima.

Nema:
- klasifikatora;
- klasicnog prediktivnog regresora;
- nasumicnog generisanja;
- uzorkovanja merenja;
- kozmetickog filtriranja rezultata.
"""


import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import scipy
import qiskit

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    DiagonalGate,
    RXGate,
    RYGate,
    RZZGate,
)
from qiskit.quantum_info import Statevector


VERSION = "QUANTUM_CAIRO_V1_20261030"


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBERS = 39
PICKS = 7

MODE_QUBITS = 6
MODES = 1 << MODE_QUBITS
CIRCUIT_LAYERS = 3

GEOMETRY_SCALES = (0.75, 1.50)
VALIDATION_TIE = 1.0e-5

RIDGE = 0.02
ANGLE_BOUND = math.pi

MAX_ITERATIONS = 80
MAX_EVALUATIONS = 120

GRADIENT_LIMIT = 5.0e-5

CHECKPOINT_EVERY = 3
PROGRESS_EVERY = 5

SCAN_BLOCK = 4096
SCAN_CHECKPOINT_EVERY = 200

PRIMES = np.array(
    [2, 3, 5, 7, 11, 13, 17],
    dtype=np.float64,
)

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

CACHE_DIR = SCRIPT_DIR / "Quantum_Cairo_V1_checkpoints"


def say(*values):
    print(*values, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: nekonacan rezultat.",
    )


def next_power_of_two(number):
    require(number >= 1, "Broj uzoraka mora biti pozitivan.")
    return 1 << (int(number) - 1).bit_length()


def hash_items(*items):
    digest = hashlib.sha256()

    for item in items:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(item).encode("utf-8"))

    return digest.hexdigest()


def atomic_npz(path, **values):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def maximum_run(numbers):
    longest = 1
    current = 1

    for previous, value in zip(numbers[:-1], numbers[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


class Geometry:
    """
    Momentna kriva:
        xi_j = (t_j, t_j^2, ..., t_j^7).

    Za sedam izabranih brojeva, prvih sedam stepenih suma
    odredjuje podskup u tacnoj aritmetici preko Newtonovih
    identiteta.

    Celobrojni potpisi cuvaju se posebno.

    Jedinstvena geometrijska koordinata ne garantuje
    razlicitu naucenu verovatnocu svake kombinacije.
    """

    def __init__(
        self,
        scale,
        numbers=NUMBERS,
        picks=PICKS,
        modes=MODES,
    ):
        self.scale = float(scale)
        self.numbers = int(numbers)
        self.picks = int(picks)
        self.modes = int(modes)

        require(
            self.picks == 7
            and self.numbers >= self.picks
            and self.modes >= 4
            and self.modes & (self.modes - 1) == 0,
            "Neispravna konfiguracija geometrije.",
        )

        self.mode_qubits = self.modes.bit_length() - 1
        self.count = math.comb(self.numbers, self.picks)

        labels = np.arange(
            1, self.numbers + 1, dtype=np.int64
        )

        self.integer_features = np.column_stack(
            [labels ** power for power in range(1, 8)]
        )

        require(
            int(np.max(self.integer_features))
            <= np.iinfo(np.int64).max // self.picks,
            "Celobrojni geometrijski potpis moze prekoraciti opseg.",
        )

        denominators = np.array(
            [float(self.numbers ** power) for power in range(1, 8)]
        )

        self.features = (
            self.integer_features.astype(np.float64)
            / denominators[None, :]
        )

        frequencies = np.zeros(
            (self.modes, 7), dtype=np.float64
        )

        # Nulti mod je konstantan.
        # Ostali modovi koriste deterministicke znakove
        # i tri dijadicke skale.
        for mode in range(1, self.modes):
            for coordinate in range(7):
                bit = (
                    mode
                    >> (coordinate % self.mode_qubits)
                ) & 1

                sign = 1.0 if bit else -1.0
                dyadic = 2.0 ** ((mode + coordinate) % 3)

                frequencies[mode, coordinate] = (
                    self.scale
                    * math.pi
                    * sign
                    * dyadic
                    * math.sqrt(PRIMES[coordinate] / 7.0)
                )

        self.frequencies = frequencies

        phase = self.features @ frequencies.T

        # B[j,r] je doprinos broja j talasnom modu r.
        self.characters = np.exp(1j * phase)

        finite(self.characters, "Geometrijski karakteri")

        require(
            np.max(
                np.abs(np.abs(self.characters) - 1.0)
            ) < 2.0e-12,
            "Karakteri nisu jedinicnog modula.",
        )

        self._build_gram()
        self._inclusion_grams = None
        self._four_indices = None
        self._four_characters = None

        minimum_separation = math.inf

        for first, second in combinations(
            range(self.numbers), 2
        ):
            overlap = np.mean(
                np.conj(self.characters[first])
                * self.characters[second]
            )

            separation = 1.0 - abs(overlap) ** 2
            minimum_separation = min(
                minimum_separation, float(separation)
            )

        self.minimum_single_swap_separation = (
            minimum_separation
        )

        require(
            minimum_separation > 1.0e-8,
            "Geometrija ne razlikuje neke zamene pojedinacnih brojeva.",
        )

    def _build_gram(self):
        """
        G[r,s] = prosek_S conj(v_r(S)) * v_s(S).

        v_r(S) = product_{j in S} B[j,r].
        """

        coefficients = np.zeros(
            (self.picks + 1, self.modes, self.modes),
            dtype=np.complex128,
        )

        coefficients[0, :, :] = 1.0

        for j in range(self.numbers):
            factors = (
                np.conj(self.characters[j, :, None])
                * self.characters[j, None, :]
            )

            upper = min(self.picks, j + 1)

            for degree in range(upper, 0, -1):
                coefficients[degree] += (
                    factors * coefficients[degree - 1]
                )

        gram = coefficients[self.picks] / self.count

        hermitian_error = float(
            np.max(np.abs(gram - gram.conj().T))
        )

        require(
            hermitian_error < 2.0e-10,
            "Gramova matrica nije hermitska.",
        )

        # Uklanjanje iskljucivo zaokruzne antisimetricne komponente.
        gram = 0.5 * (gram + gram.conj().T)

        finite(gram, "Gramova matrica")

        require(
            np.max(np.abs(np.diag(gram) - 1.0))
            < 2.0e-10,
            "Dijagonala Gramove matrice nije jedan.",
        )

        eigenvalues = np.linalg.eigvalsh(gram)

        require(
            np.min(eigenvalues) >= -2.0e-9,
            "Gramova matrica nije pozitivno semidefinitna.",
        )

        self.gram = gram
        self.coefficients = coefficients
        self.minimum_eigenvalue = float(np.min(eigenvalues))

    def subset_characters(self, draws):
        draws = np.asarray(draws, dtype=np.int64)

        require(
            draws.ndim == 2
            and draws.shape[1] == self.picks
            and np.all(draws >= 0)
            and np.all(draws < self.numbers)
            and np.all(np.diff(draws, axis=1) > 0),
            "Neispravan podskup za geometriju.",
        )

        result = np.ones(
            (len(draws), self.modes),
            dtype=np.complex128,
        )

        for position in range(self.picks):
            result *= self.characters[draws[:, position]]

        finite(result, "Karakteri kombinacija")
        return result

    def mean_weight(self, amplitudes):
        amplitudes = np.asarray(
            amplitudes, dtype=np.complex128
        )

        transformed = amplitudes @ self.gram.T

        values = np.sum(
            np.conj(amplitudes) * transformed,
            axis=1,
        )

        require(
            np.max(np.abs(values.imag)) < 2.0e-8,
            "Normalizaciona kvadratna forma nije realna.",
        )

        result = values.real

        finite(result, "Srednja tezina raspodele")

        require(
            np.all(result > 0.0),
            "Kandidat ima nepozitivnu normalizaciju.",
        )

        return result, transformed

    def inclusion_grams(self):
        if self._inclusion_grams is not None:
            return self._inclusion_grams

        matrices = np.empty(
            (self.numbers, self.modes, self.modes),
            dtype=np.complex128,
        )

        for j in range(self.numbers):
            factor = (
                np.conj(self.characters[j, :, None])
                * self.characters[j, None, :]
            )

            # Sinteticko deljenje generisuceg polinoma:
            # E_without_j[k] = E_total[k] - factor_j * E_without_j[k-1].
            excluded = np.ones(
                (self.modes, self.modes),
                dtype=np.complex128,
            )

            for degree in range(1, self.picks):
                excluded = (
                    self.coefficients[degree]
                    - factor * excluded
                )

            matrix = factor * excluded / self.count
            matrices[j] = 0.5 * (
                matrix + matrix.conj().T
            )

        finite(matrices, "Matrice ukljucivanja")

        require(
            np.max(
                np.abs(
                    matrices.sum(axis=0)
                    - self.picks * self.gram
                )
            ) < 2.0e-8,
            "Matrice ukljucivanja nemaju ispravan zbir.",
        )

        self._inclusion_grams = matrices
        return matrices

    def inclusion_probabilities(self, amplitudes):
        amplitudes = np.asarray(
            amplitudes, dtype=np.complex128
        )

        mean_weight, _ = self.mean_weight(amplitudes)

        values = np.einsum(
            "nr,jrs,ns->nj",
            np.conj(amplitudes),
            self.inclusion_grams(),
            amplitudes,
            optimize=True,
        )

        require(
            np.max(np.abs(values.imag)) < 2.0e-8,
            "Verovatnoce ukljucivanja nisu realne.",
        )

        probabilities = (
            values.real / mean_weight[:, None]
        )

        finite(probabilities, "Verovatnoce ukljucivanja")

        require(
            np.min(probabilities) >= -2.0e-8
            and np.max(probabilities) <= 1.0 + 2.0e-8,
            "Verovatnoce ukljucivanja su izvan dozvoljenog opsega.",
        )

        require(
            np.max(
                np.abs(
                    probabilities.sum(axis=1) - self.picks
                )
            ) < 2.0e-7,
            "Zbir verovatnoca ukljucivanja nije sedam.",
        )

        return probabilities

    def prepare_four_table(self):
        if self._four_indices is not None:
            return self._four_indices, self._four_characters

        indices = np.array(
            list(combinations(range(self.numbers), 4)),
            dtype=np.int16,
        )

        values = np.ones(
            (len(indices), self.modes),
            dtype=np.complex128,
        )

        for position in range(4):
            values *= self.characters[indices[:, position]]

        finite(values, "Tabela cetvoroclanih podskupova")

        self._four_indices = indices
        self._four_characters = values

        return indices, values


def make_history_inputs(all_characters):
    """
    Ulaz za predvidjanje posle kola t koristi samo:
    t, t-1 i t-2.

    Koristi koherentne karaktere celih prethodnih kombinacija.
    Ne koristi rangiranje ucestalosti pojedinacnih brojeva.
    """

    all_characters = np.asarray(
        all_characters, dtype=np.complex128
    )

    rows = np.arange(len(all_characters))

    inputs = (
        np.conj(all_characters[rows])
        + 0.6 * np.conj(
            all_characters[np.maximum(rows - 1, 0)]
        )
        + 0.3 * np.conj(
            all_characters[np.maximum(rows - 2, 0)]
        )
    )

    norms = np.sqrt(
        np.sum(np.abs(inputs) ** 2, axis=1)
    )

    require(
        np.all(norms > 0.0),
        "Istorijsko kodiranje ima nultu normu.",
    )

    inputs /= norms[:, None]

    finite(inputs, "Istorijske amplitude")
    return inputs


class QuantumBatch:
    """
    Paket nezavisnih ulaznih stanja.

    Indeksni bitovi sluze samo vektorizaciji simulatora.
    Naucene kapije deluju iskljucivo na kubite talasnih modova.
    Ovo nije tvrdnja o kvantnom ubrzanju.
    """

    def __init__(self, inputs):
        self.inputs = np.asarray(
            inputs, dtype=np.complex128
        )

        require(
            self.inputs.ndim == 2
            and len(self.inputs) > 0,
            "Neispravan paket amplituda.",
        )

        self.samples, self.modes = self.inputs.shape

        require(
            self.modes & (self.modes - 1) == 0,
            "Broj modova nije stepen dvojke.",
        )

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(self.inputs) ** 2,
                        axis=1,
                    ) - 1.0
                )
            ) < 2.0e-10,
            "Ulazne amplitude nisu normalizovane.",
        )

        self.padded = next_power_of_two(self.samples)

        packed = np.zeros(
            (self.padded, self.modes),
            dtype=np.complex128,
        )

        packed[:self.samples] = (
            self.inputs / math.sqrt(self.samples)
        )

        self.packed = packed.reshape(-1)


def quantum_gate(name, angle):
    if name == "rx":
        return RXGate(float(angle))

    if name == "ry":
        return RYGate(float(angle))

    if name == "rzz":
        return RZZGate(float(angle))

    raise RuntimeError("Nepoznata kvantna kapija.")


class QuantumSource:
    def __init__(
        self,
        qubits=MODE_QUBITS,
        layers=CIRCUIT_LAYERS,
    ):
        self.qubits = int(qubits)
        self.layers = int(layers)
        self.modes = 1 << self.qubits

        require(
            self.qubits >= 2 and self.layers >= 1,
            "Neispravna konfiguracija kvantnog izvora.",
        )

        self.size = self.layers * 3 * self.qubits
        self.indices = np.arange(self.modes, dtype=np.int64)

    def seed(self):
        positions = np.arange(
            self.size, dtype=np.float64
        )

        return (
            0.07 * np.sin(positions + 0.4)
            + 0.03 * np.cos(0.7 * positions)
        )

    def operations(self, parameters):
        values = np.asarray(parameters).reshape(
            self.layers, 3, self.qubits
        )

        result = []

        for layer in range(self.layers):
            base = layer * 3 * self.qubits

            for q in range(self.qubits):
                result.append(
                    (
                        "rx",
                        float(values[layer, 0, q]),
                        (q,),
                        base + q,
                    )
                )

            for q in range(self.qubits):
                result.append(
                    (
                        "rzz",
                        float(values[layer, 1, q]),
                        (q, (q + 1) % self.qubits),
                        base + self.qubits + q,
                    )
                )

            for q in range(self.qubits):
                result.append(
                    (
                        "ry",
                        float(values[layer, 2, q]),
                        (q,),
                        base + 2 * self.qubits + q,
                    )
                )

        return result

    def circuit(self, parameters):
        circuit = QuantumCircuit(self.qubits)

        for name, angle, qubits, _ in self.operations(parameters):
            circuit.append(
                quantum_gate(name, angle),
                list(qubits),
            )

        return circuit

    def forward(self, parameters, batch):
        parameters = np.asarray(
            parameters, dtype=np.float64
        )

        require(
            parameters.shape == (self.size,),
            "Neispravan broj parametara.",
        )

        require(
            batch.modes == self.modes,
            "Paket i kvantni izvor nemaju isti broj modova.",
        )

        finite(parameters, "Kvantni parametri")

        circuit = self.circuit(parameters)
        state = Statevector(batch.packed).evolve(circuit)

        packed = np.asarray(
            state.data, dtype=np.complex128
        ).reshape(batch.padded, self.modes)

        amplitudes = (
            packed[:batch.samples]
            * math.sqrt(batch.samples)
        )

        finite(amplitudes, "Qiskit amplitude")

        require(
            np.max(
                np.abs(
                    np.sum(np.abs(amplitudes) ** 2, axis=1)
                    - 1.0
                )
            ) < 2.0e-9,
            "Qiskit kolo nije sacuvalo normu.",
        )

        if batch.padded > batch.samples:
            require(
                np.max(
                    np.abs(packed[batch.samples:])
                ) < 2.0e-12,
                "Promenjene su prazne indeksne grane.",
            )

        return amplitudes, np.array(
            state.data, copy=True
        )

    def generator(self, vector, name, qubits):
        matrix = np.asarray(vector).reshape(
            -1, self.modes
        )

        q = qubits[0]

        if name == "rx":
            return matrix[
                :, self.indices ^ (1 << q)
            ].reshape(-1)

        if name == "ry":
            bit = ((self.indices >> q) & 1).astype(
                np.float64
            )

            signs = 1j * (2.0 * bit - 1.0)

            return (
                matrix[:, self.indices ^ (1 << q)]
                * signs[None, :]
            ).reshape(-1)

        if name == "rzz":
            r = qubits[1]

            parity = (
                ((self.indices >> q) & 1)
                ^ ((self.indices >> r) & 1)
            )

            signs = 1.0 - 2.0 * parity

            return (
                matrix * signs[None, :]
            ).reshape(-1)

        raise RuntimeError("Nepoznat generator.")

    def gradient(
        self,
        parameters,
        batch,
        final_state,
        amplitude_derivative,
    ):
        """
        Adjungovani gradijent srednjeg NLL-a.

        Vektor adjoint je matematicki izvod.
        Ne predstavlja dodatno kvantno stanje modela.
        """

        derivative = np.zeros(
            (batch.padded, self.modes),
            dtype=np.complex128,
        )

        derivative[:batch.samples] = (
            amplitude_derivative / math.sqrt(batch.samples)
        )

        state = Statevector(final_state)
        adjoint = Statevector(derivative.reshape(-1))

        gradient = np.zeros(
            self.size, dtype=np.float64
        )

        for name, angle, qubits, index in reversed(
            self.operations(parameters)
        ):
            generated = self.generator(
                state.data, name, qubits
            )

            gradient[index] = float(
                np.real(
                    -1j * np.vdot(
                        adjoint.data, generated
                    )
                )
            )

            inverse = quantum_gate(name, -angle)

            state = state.evolve(
                inverse, qargs=list(qubits)
            )

            adjoint = adjoint.evolve(
                inverse, qargs=list(qubits)
            )

        finite(gradient, "Adjungovani gradijent")
        return gradient


def objective(
    parameters,
    source,
    geometry,
    batch,
    observed_characters,
):
    amplitudes, final_state = source.forward(
        parameters, batch
    )

    mean_weight, transformed = geometry.mean_weight(
        amplitudes
    )

    observed_amplitude = np.sum(
        amplitudes * observed_characters,
        axis=1,
    )

    observed_weight = np.abs(observed_amplitude) ** 2

    require(
        np.all(observed_weight > 0.0),
        "Kandidat ima nultu tezinu istorijskog cilja.",
    )

    log_probability = (
        np.log(observed_weight)
        - np.log(mean_weight)
        - math.log(geometry.count)
    )

    finite(log_probability, "Logaritmi verovatnoca")

    require(
        np.max(log_probability) <= 2.0e-7,
        "Verovatnoca istorijske kombinacije prelazi jedan.",
    )

    nll = -float(np.mean(log_probability))

    amplitude_derivative = (
        transformed / mean_weight[:, None]
        - np.conj(observed_characters)
        * (
            observed_amplitude / observed_weight
        )[:, None]
    )

    gradient = source.gradient(
        parameters,
        batch,
        final_state,
        amplitude_derivative,
    )

    regularization = 0.5 * RIDGE * float(
        np.mean(parameters ** 2)
    )

    gradient += RIDGE * parameters / source.size

    loss = nll + regularization

    require(
        math.isfinite(loss),
        "Funkcija cilja nije konacna.",
    )

    finite(gradient, "Ukupan gradijent")

    return loss, gradient


def projected_gradient(parameters, gradient):
    result = np.asarray(gradient).copy()

    at_lower = parameters <= -ANGLE_BOUND + 1.0e-10
    at_upper = parameters >= ANGLE_BOUND - 1.0e-10

    result[at_lower & (gradient > 0.0)] = 0.0
    result[at_upper & (gradient < 0.0)] = 0.0

    return float(np.max(np.abs(result)))


@dataclass
class Fit:
    parameters: np.ndarray
    gradient: float
    loss: float
    message: str


class EvaluationBudgetReached(Exception):
    pass


def save_fit(
    path,
    parameters,
    iterations,
    done,
    message,
):
    atomic_npz(
        path,
        version=np.array(VERSION),
        parameters=np.asarray(
            parameters, dtype=np.float64
        ),
        iterations=np.array(iterations, dtype=np.int64),
        done=np.array(bool(done)),
        message=np.array(str(message)),
    )


def fit_model(
    source,
    geometry,
    inputs,
    observed_characters,
    label,
    warm=None,
):
    batch = QuantumBatch(inputs)

    initial = (
        source.seed()
        if warm is None
        else warm.parameters.copy()
    )

    key = hash_items(
        VERSION,
        label,
        geometry.scale,
        source.qubits,
        source.layers,
        RIDGE,
        ANGLE_BOUND,
        MAX_ITERATIONS,
        MAX_EVALUATIONS,
        np.__version__,
        scipy.__version__,
        qiskit.__version__,
        geometry.characters,
        inputs,
        observed_characters,
        initial,
    )

    path = CACHE_DIR / ("fit_" + key + ".npz")

    parameters = initial.copy()
    iterations = 0
    done = False
    message = "Pocetak."

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    str(saved["version"].item()) == VERSION,
                    "Neodgovarajuca verzija kontrolne tacke.",
                )

                parameters = np.asarray(
                    saved["parameters"],
                    dtype=np.float64,
                ).copy()

                iterations = int(
                    saved["iterations"].item()
                )

                done = bool(saved["done"].item())
                message = str(saved["message"].item())

            require(
                parameters.shape == (source.size,),
                "Neispravan oblik sacuvanih parametara.",
            )

            finite(parameters, "Sacuvani parametri")

            require(
                np.max(np.abs(parameters))
                <= ANGLE_BOUND + 1.0e-10,
                "Sacuvani parametri su izvan granica.",
            )

            require(
                0 <= iterations <= MAX_ITERATIONS,
                "Neispravan sacuvani broj iteracija.",
            )

            say(label, "| ucitana kontrolna tacka.")

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            say(
                label,
                "| kontrolna tacka nije prihvacena:",
                str(error),
            )

            parameters = initial.copy()
            iterations = 0
            done = False
            message = "Ponovljen pocetak."

    start = time.perf_counter()

    loss, gradient = objective(
        parameters,
        source,
        geometry,
        batch,
        observed_characters,
    )

    best_parameters = parameters.copy()
    best_loss = float(loss)
    best_gradient = gradient.copy()

    calls = 0

    say()
    say(label)
    say(
        f"Skala: {geometry.scale:.2f} "
        f"| uzorci: {len(inputs)} "
        f"| parametri: {source.size}"
    )

    def function(candidate):
        nonlocal calls
        nonlocal best_parameters, best_loss, best_gradient

        if calls >= MAX_EVALUATIONS:
            raise EvaluationBudgetReached()

        calls += 1

        value, derivative = objective(
            candidate,
            source,
            geometry,
            batch,
            observed_characters,
        )

        if value < best_loss:
            best_loss = float(value)
            best_parameters = np.array(
                candidate, copy=True
            )
            best_gradient = np.array(
                derivative, copy=True
            )

        if calls == 1 or calls % PROGRESS_EVERY == 0:
            say(
                f"Evaluacija {calls}/{MAX_EVALUATIONS} "
                f"| cilj {best_loss:.9f} "
                f"| gradijent "
                f"{projected_gradient(best_parameters, best_gradient):.3e} "
                f"| proteklo "
                f"{time.perf_counter() - start:.1f} s"
            )

        return value, derivative

    def callback(_candidate):
        nonlocal iterations

        iterations += 1

        if iterations % CHECKPOINT_EVERY == 0:
            save_fit(
                path,
                best_parameters,
                iterations,
                False,
                "Optimizacija u toku.",
            )

    try:
        if not done and iterations < MAX_ITERATIONS:
            result = minimize(
                function,
                best_parameters,
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (-ANGLE_BOUND, ANGLE_BOUND)
                ] * source.size,
                callback=callback,
                options={
                    "maxiter": MAX_ITERATIONS - iterations,
                    "maxfun": MAX_EVALUATIONS,
                    "maxls": 20,
                    "maxcor": 12,
                    "ftol": 1.0e-11,
                    "gtol": 1.0e-6,
                },
            )

            message = str(result.message)

        elif not done:
            message = "Dostignut budzet iteracija."

    except EvaluationBudgetReached:
        message = "Dostignut budzet evaluacija."

    except KeyboardInterrupt:
        save_fit(
            path,
            best_parameters,
            iterations,
            False,
            "Korisnicki prekid.",
        )
        raise

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        message = (
            "Zadrzan najbolji prethodni numericki vazeci kandidat. "
            + str(error)
        )

    checked_loss, checked_gradient = objective(
        best_parameters,
        source,
        geometry,
        batch,
        observed_characters,
    )

    require(
        abs(checked_loss - best_loss)
        < 2.0e-8 * (1.0 + abs(best_loss)),
        "Ponovljeni proracun funkcije cilja se ne slaze.",
    )

    gradient_norm = projected_gradient(
        best_parameters, checked_gradient
    )

    save_fit(
        path,
        best_parameters,
        min(iterations, MAX_ITERATIONS),
        True,
        message,
    )

    say(f"Najbolji cilj: {checked_loss:.9f}")
    say(f"Max|projektovani gradijent|: {gradient_norm:.3e}")
    say("Optimizator:", message)

    if gradient_norm > GRADIENT_LIMIT:
        say(
            "DIJAGNOSTIKA: stacionarnost nije potvrdjena. "
            "Budzet nije razlog za odbacivanje vazeceg kandidata."
        )

    return Fit(
        parameters=best_parameters,
        gradient=gradient_norm,
        loss=float(checked_loss),
        message=message,
    )


def predict_amplitudes(source, fit, inputs):
    amplitudes, _ = source.forward(
        fit.parameters, QuantumBatch(inputs)
    )
    return amplitudes


def evaluate(
    source,
    fit,
    geometry,
    inputs,
    observed_characters,
):
    amplitudes = predict_amplitudes(
        source, fit, inputs
    )

    mean_weight, _ = geometry.mean_weight(amplitudes)

    observed_weight = np.abs(
        np.sum(
            amplitudes * observed_characters,
            axis=1,
        )
    ) ** 2

    require(
        np.all(observed_weight > 0.0),
        "Evaluacija ima nultu tezinu cilja.",
    )

    log_probability = (
        np.log(observed_weight)
        - np.log(mean_weight)
        - math.log(geometry.count)
    )

    finite(log_probability, "Evaluaciona verovatnoca")

    require(
        np.max(log_probability) <= 2.0e-7,
        "Evaluaciona verovatnoca prelazi jedan.",
    )

    return -float(np.mean(log_probability))


def exhaustive_next(
    geometry,
    amplitudes,
    checkpoint=None,
    verbose=True,
):
    """
    Tacna deterministicka pretraga:
    kombinacija = prva tri broja + poslednja cetiri broja.

    Prva tri su manja od svakog od poslednja cetiri.
    Svaka kombinacija pojavljuje se tacno jednom.
    """

    amplitudes = np.asarray(
        amplitudes, dtype=np.complex128
    )

    require(
        amplitudes.shape == (geometry.modes,),
        "Neispravne NEXT amplitude.",
    )

    four_indices, four_values = (
        geometry.prepare_four_table()
    )

    prefixes = list(
        combinations(
            range(geometry.numbers - 4),
            3,
        )
    )

    completed = 0
    visited = 0

    total_weight = np.longdouble(0.0)

    inclusion_weight = np.zeros(
        geometry.numbers,
        dtype=np.longdouble,
    )

    best_weight = -1.0
    best_selected = np.zeros(
        geometry.picks, dtype=np.int64
    )

    if checkpoint is not None and checkpoint.exists():
        try:
            with np.load(
                checkpoint, allow_pickle=False
            ) as saved:
                require(
                    str(saved["version"].item()) == VERSION,
                    "Neodgovarajuca verzija pretrage.",
                )

                completed = int(saved["completed"].item())
                visited = int(saved["visited"].item())

                total_weight = np.longdouble(
                    saved["total_weight"].item()
                )

                inclusion_weight = np.asarray(
                    saved["inclusion_weight"],
                    dtype=np.longdouble,
                ).copy()

                best_weight = float(
                    saved["best_weight"].item()
                )

                best_selected = np.asarray(
                    saved["best_selected"],
                    dtype=np.int64,
                ).copy()

            require(
                0 <= completed <= len(prefixes)
                and 0 <= visited <= geometry.count
                and inclusion_weight.shape
                == (geometry.numbers,)
                and best_selected.shape
                == (geometry.picks,),
                "Neispravna kontrolna tacka pretrage.",
            )

            finite(
                inclusion_weight,
                "Sacuvane tezine ukljucivanja",
            )

            require(
                np.isfinite(total_weight)
                and total_weight >= 0.0,
                "Neispravna sacuvana masa.",
            )

            if verbose:
                say(
                    "Nastavljam pretragu od pregledanih:",
                    f"{visited:,}",
                )

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            if verbose:
                say(
                    "Kontrolna tacka pretrage nije prihvacena:",
                    str(error),
                )

            completed = 0
            visited = 0
            total_weight = np.longdouble(0.0)

            inclusion_weight = np.zeros(
                geometry.numbers,
                dtype=np.longdouble,
            )

            best_weight = -1.0
            best_selected = np.zeros(
                geometry.picks, dtype=np.int64
            )

    start = time.perf_counter()

    def save_progress():
        if checkpoint is None:
            return

        atomic_npz(
            checkpoint,
            version=np.array(VERSION),
            completed=np.array(completed, dtype=np.int64),
            visited=np.array(visited, dtype=np.int64),
            total_weight=np.array(
                total_weight, dtype=np.longdouble
            ),
            inclusion_weight=inclusion_weight,
            best_weight=np.array(best_weight),
            best_selected=best_selected,
        )

    # Prekid se cuva samo na granicama potpuno obradjenih prefiksa.
    # Ako se prekid desi unutar prefiksa, njegova delimicna masa
    # se ne upisuje preko prethodne kontrolne tacke.
    for prefix_index in range(completed, len(prefixes)):
        prefix = prefixes[prefix_index]
        last_prefix = prefix[-1]

        begin = int(
            np.searchsorted(
                four_indices[:, 0],
                last_prefix,
                side="right",
            )
        )

        prefix_factor = amplitudes.copy()

        for number in prefix:
            prefix_factor *= geometry.characters[number]

        prefix_mass = np.longdouble(0.0)

        prefix_inclusion = np.zeros(
            geometry.numbers,
            dtype=np.longdouble,
        )

        prefix_visited = 0
        prefix_best_weight = best_weight
        prefix_best_selected = best_selected.copy()

        for block_start in range(
            begin, len(four_indices), SCAN_BLOCK
        ):
            block_end = min(
                block_start + SCAN_BLOCK,
                len(four_indices),
            )

            selected_four = four_indices[
                block_start:block_end
            ]

            combined_amplitude = (
                four_values[block_start:block_end]
                @ prefix_factor
            )

            weights = np.abs(combined_amplitude) ** 2
            finite(weights, "Tezine tokom globalne pretrage")

            require(
                np.max(weights)
                <= geometry.modes + 2.0e-8,
                "Koherentna tezina prelazi Cauchyjevu granicu.",
            )

            mass = np.sum(
                weights, dtype=np.longdouble
            )

            prefix_mass += mass
            prefix_visited += len(weights)

            for position in range(4):
                counts = np.bincount(
                    selected_four[:, position],
                    weights=weights,
                    minlength=geometry.numbers,
                )

                prefix_inclusion += counts.astype(
                    np.longdouble
                )

            block_maximum = int(np.argmax(weights))
            candidate_weight = float(
                weights[block_maximum]
            )

            if candidate_weight > prefix_best_weight:
                prefix_best_weight = candidate_weight

                prefix_best_selected = np.array(
                    prefix
                    + tuple(
                        int(value)
                        for value in selected_four[block_maximum]
                    ),
                    dtype=np.int64,
                )

        for number in prefix:
            prefix_inclusion[number] += prefix_mass

        total_weight += prefix_mass
        inclusion_weight += prefix_inclusion
        visited += prefix_visited

        best_weight = prefix_best_weight
        best_selected = prefix_best_selected

        completed = prefix_index + 1

        should_save = (
            completed % SCAN_CHECKPOINT_EVERY == 0
            or completed == len(prefixes)
        )

        if should_save:
            save_progress()

            if verbose:
                say(
                    f"Pregledano {visited:,}/{geometry.count:,} "
                    f"| {100.0 * visited / geometry.count:.2f}% "
                    f"| proteklo "
                    f"{time.perf_counter() - start:.1f} s"
                )

    require(
        visited == geometry.count,
        "Globalna pretraga nije obuhvatila sve kombinacije.",
    )

    require(
        total_weight > 0.0 and best_weight > 0.0,
        "Globalna pretraga nema pozitivnu masu.",
    )

    require(
        np.all(np.diff(best_selected) > 0),
        "Globalni maksimum nije legalna kombinacija.",
    )

    inclusion = np.asarray(
        inclusion_weight / total_weight,
        dtype=np.float64,
    )

    require(
        np.min(inclusion) >= 0.0
        and np.max(inclusion) <= 1.0 + 2.0e-9,
        "Pretraga daje neispravne verovatnoce ukljucivanja.",
    )

    require(
        abs(float(inclusion.sum()) - geometry.picks)
        < 2.0e-7,
        "Pretraga daje neispravan zbir ukljucivanja.",
    )

    return {
        "selected": best_selected,
        "weight": best_weight,
        "total_weight": total_weight,
        "inclusion": inclusion,
        "visited": visited,
    }


def qiskit_interference_check(
    source,
    fit,
    next_input,
    geometry,
    selected,
    expected_amplitudes,
):
    direct = Statevector(next_input).evolve(
        source.circuit(fit.parameters)
    )

    source_error = float(
        np.max(
            np.abs(
                direct.data - expected_amplitudes
            )
        )
    )

    characters = geometry.subset_characters(
        np.asarray(selected, dtype=np.int64)[None, :]
    )[0]

    circuit = QuantumCircuit(source.qubits)

    circuit.append(
        DiagonalGate(characters.tolist()),
        list(range(source.qubits)),
    )

    for q in range(source.qubits):
        circuit.h(q)

    final = direct.evolve(circuit)

    actual = float(abs(final.data[0]) ** 2)

    expected = float(
        abs(
            np.sum(
                expected_amplitudes * characters
            )
        ) ** 2 / geometry.modes
    )

    interference_error = abs(actual - expected)

    require(
        source_error < 2.0e-10
        and interference_error < 2.0e-10,
        "Eksplicitna Qiskit provera interferencije nije prosla.",
    )

    return source_error, interference_error, actual


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    blanks = 0

    for line_number, row in enumerate(
        csv.reader(
            raw.decode("utf-8-sig").splitlines()
        ),
        start=1,
    ):
        if not row or all(
            not value.strip() for value in row
        ):
            blanks += 1
            continue

        require(
            len(row) == PICKS,
            f"{path.name}, red {line_number}: "
            "potrebno je sedam kolona.",
        )

        try:
            values = [
                int(value.strip()) for value in row
            ]

        except ValueError as error:
            raise RuntimeError(
                f"{path.name}, red {line_number}: "
                "vrednosti moraju biti celi brojevi."
            ) from error

        require(
            len(set(values)) == PICKS
            and min(values) >= 1
            and max(values) <= NUMBERS,
            f"{path.name}, red {line_number}: "
            "neispravna kombinacija.",
        )

        rows.append(
            sorted(value - 1 for value in values)
        )

    require(
        len(rows) >= 100,
        f"{path.name}: potrebno je najmanje 100 kola.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    say()
    say("ULAZ:", path.name)
    say(f"Redova: {len(draws)} | praznih redova: {blanks}")
    say("SHA256:", fingerprint)
    say("Redosled kola ostaje redosled iz CSV-a.")

    return draws, fingerprint


def self_checks():
    say()
    say("Pokrecem matematicke i Qiskit samoprovere.")

    geometry = Geometry(
        0.75,
        numbers=9,
        picks=7,
        modes=8,
    )

    subsets = np.array(
        list(combinations(range(9), 7)),
        dtype=np.int64,
    )

    characters = geometry.subset_characters(subsets)

    direct_gram = (
        characters.conj().T @ characters
        / len(subsets)
    )

    require(
        np.max(
            np.abs(
                direct_gram - geometry.gram
            )
        ) < 2.0e-11,
        "Gramova matrica ne prolazi potpuno nabrajanje.",
    )

    signatures = (
        geometry.integer_features[subsets].sum(axis=1)
    )

    require(
        len(
            {
                tuple(int(value) for value in signature)
                for signature in signatures
            }
        ) == len(subsets),
        "Mali geometrijski primer ima koliziju potpisa.",
    )

    raw = (
        1.0 + 0.2 * np.sin(np.arange(8) + 0.3)
    ) * np.exp(0.17j * np.arange(8))

    amplitude = raw / np.linalg.norm(raw)

    weights = np.abs(characters @ amplitude) ** 2

    mean_weight, _ = geometry.mean_weight(
        amplitude[None, :]
    )

    require(
        abs(
            float(np.mean(weights))
            - mean_weight[0]
        ) < 2.0e-11,
        "Normalizacija se ne slaze sa nabrajanjem.",
    )

    direct_inclusion = np.zeros(
        geometry.numbers, dtype=np.float64
    )

    for subset, weight in zip(subsets, weights):
        direct_inclusion[subset] += weight

    direct_inclusion /= weights.sum()

    analytic_inclusion = (
        geometry.inclusion_probabilities(
            amplitude[None, :]
        )[0]
    )

    require(
        np.max(
            np.abs(
                direct_inclusion - analytic_inclusion
            )
        ) < 2.0e-10,
        "Marginalne verovatnoce nisu ispravne.",
    )

    scan = exhaustive_next(
        geometry,
        amplitude,
        checkpoint=None,
        verbose=False,
    )

    exact_maximum = int(np.argmax(weights))

    require(
        np.array_equal(
            scan["selected"],
            subsets[exact_maximum],
        ),
        "Pretraga ne nalazi tacan maksimum.",
    )

    require(
        abs(
            float(scan["total_weight"])
            - float(weights.sum())
        ) < 2.0e-10,
        "Pretraga ne daje ispravnu ukupnu masu.",
    )

    # Granica sa jednom aktivnom talasnom komponentom.
    single_mode = np.zeros(8, dtype=np.complex128)
    single_mode[0] = 1.0

    single_weights = np.abs(
        characters @ single_mode
    ) ** 2

    require(
        np.max(np.abs(single_weights - 1.0)) < 2.0e-12,
        "Granicni slucaj jednog moda nije ispravan.",
    )

    # Istorijski ulazi i gradijent malog Qiskit modela.
    history = characters[[0, 7, 18, 30]]

    inputs = make_history_inputs(history)[:3]
    observed = characters[[7, 18, 30]]

    source = QuantumSource(qubits=3, layers=2)
    parameters = source.seed()

    batch = QuantumBatch(inputs)

    loss, analytic_gradient = objective(
        parameters,
        source,
        geometry,
        batch,
        observed,
    )

    require(
        math.isfinite(loss),
        "Samoprovera funkcije cilja nije konacna.",
    )

    epsilon = 2.0e-6
    maximum_gradient_error = 0.0

    for index in range(source.size):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        loss_plus = objective(
            plus,
            source,
            geometry,
            batch,
            observed,
        )[0]

        loss_minus = objective(
            minus,
            source,
            geometry,
            batch,
            observed,
        )[0]

        numerical = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

        error = abs(
            numerical - analytic_gradient[index]
        )

        maximum_gradient_error = max(
            maximum_gradient_error, error
        )

        require(
            error
            < 2.0e-5 * (
                1.0 + abs(analytic_gradient[index])
            ),
            "Qiskit adjungovani gradijent nije ispravan.",
        )

    batch_amplitudes, _ = source.forward(
        parameters, batch
    )

    maximum_batch_error = 0.0

    for row in range(len(inputs)):
        direct = Statevector(inputs[row]).evolve(
            source.circuit(parameters)
        )

        error = float(
            np.max(
                np.abs(
                    direct.data - batch_amplitudes[row]
                )
            )
        )

        maximum_batch_error = max(
            maximum_batch_error, error
        )

        require(
            error < 2.0e-11,
            "Paketna i pojedinacna simulacija se razlikuju.",
        )

    test_fit = Fit(
        parameters=parameters,
        gradient=0.0,
        loss=loss,
        message="Samoprovera.",
    )

    _, interference_error, _ = qiskit_interference_check(
        source,
        test_fit,
        inputs[0],
        geometry,
        subsets[4],
        batch_amplitudes[0],
    )

    changed_parameters = parameters.copy()
    changed_parameters[-1] += 0.19

    changed_amplitudes, _ = source.forward(
        changed_parameters, batch
    )

    original_weights = np.abs(
        characters @ batch_amplitudes[0]
    ) ** 2

    changed_weights = np.abs(
        characters @ changed_amplitudes[0]
    ) ** 2

    original_probabilities = (
        original_weights / original_weights.sum()
    )

    changed_probabilities = (
        changed_weights / changed_weights.sum()
    )

    require(
        np.max(
            np.abs(
                changed_probabilities
                - original_probabilities
            )
        ) > 1.0e-8,
        "Parametri ne menjaju zajednicku raspodelu.",
    )

    say("Samoprovere su prosle.")
    say(
        f"Najveca greska gradijenta: "
        f"{maximum_gradient_error:.3e}"
    )
    say(
        f"Najveca greska paketne simulacije: "
        f"{maximum_batch_error:.3e}"
    )
    say(
        f"Greska eksplicitne interferencije: "
        f"{interference_error:.3e}"
    )


def process_file(
    path,
    draws,
    fingerprint,
    source,
    geometries,
):
    count = len(draws) - 1

    cut60 = int(0.60 * count)
    cut70 = int(0.70 * count)
    cut80 = int(0.80 * count)

    folds = (
        (cut60, cut70),
        (cut70, cut80),
    )

    candidates = []
    prepared = {}

    say()
    say("=" * 70)
    say("OBRADA:", path.name)
    say(f"Istorijskih prelaza: {count}")

    say(
        "Validacija: 60%-70% i 70%-80%; "
        "odvojeni test: poslednjih 20% prelaza."
    )

    for scale in GEOMETRY_SCALES:
        geometry = geometries[scale]

        all_characters = geometry.subset_characters(draws)
        all_inputs = make_history_inputs(all_characters)

        inputs = all_inputs[:-1]
        observed = all_characters[1:]

        prepared[scale] = (
            all_inputs,
            all_characters,
        )

        weighted_nll = 0.0
        validation_rows = 0
        warm = None

        for fold_index, (train_end, valid_end) in enumerate(folds):
            fit = fit_model(
                source,
                geometry,
                inputs[:train_end],
                observed[:train_end],
                (
                    f"{path.name} | skala {scale:.2f} "
                    f"| validacija {fold_index + 1}"
                ),
                warm=warm,
            )

            score = evaluate(
                source,
                fit,
                geometry,
                inputs[train_end:valid_end],
                observed[train_end:valid_end],
            )

            rows = valid_end - train_end

            weighted_nll += score * rows
            validation_rows += rows
            warm = fit

            say(f"Validacioni NLL: {score:.9f}")

        candidate = {
            "scale": scale,
            "validation": weighted_nll / validation_rows,
            "warm": warm,
        }

        candidates.append(candidate)

        say()
        say(
            f"Skala {scale:.2f} "
            f"| objedinjeni validacioni NLL: "
            f"{candidate['validation']:.9f}"
        )

    best_score = min(
        candidate["validation"]
        for candidate in candidates
    )

    equivalent = [
        candidate
        for candidate in candidates
        if candidate["validation"]
        <= best_score + VALIDATION_TIE
    ]

    selected = min(
        equivalent,
        key=lambda candidate: candidate["scale"],
    )

    scale = selected["scale"]
    geometry = geometries[scale]

    all_inputs, all_characters = prepared[scale]

    inputs = all_inputs[:-1]
    observed = all_characters[1:]
    next_input = all_inputs[-1:]

    say()
    say("IZBOR GEOMETRIJSKE SKALE")

    for candidate in candidates:
        say(
            f"Skala {candidate['scale']:.2f} "
            f"| validacioni NLL "
            f"{candidate['validation']:.9f}"
        )

    say(f"Izabrana skala: {scale:.2f}")
    say(
        f"Tolerancija pri izboru manje skale: "
        f"{VALIDATION_TIE:.6f} NLL"
    )

    test_fit = fit_model(
        source,
        geometry,
        inputs[:cut80],
        observed[:cut80],
        f"{path.name} | odvojeni test",
        warm=selected["warm"],
    )

    test_nll = evaluate(
        source,
        test_fit,
        geometry,
        inputs[cut80:],
        observed[cut80:],
    )

    say()
    say(f"Odvojeni test NLL: {test_nll:.9f}")
    say(
        "Test ne menja izabranu skalu."
    )

    final_fit = fit_model(
        source,
        geometry,
        inputs,
        observed,
        f"{path.name} | zavrsni trening cele istorije",
        warm=test_fit,
    )

    next_amplitudes = predict_amplitudes(
        source,
        final_fit,
        next_input,
    )[0]

    mean_weight, _ = geometry.mean_weight(
        next_amplitudes[None, :]
    )

    analytic_total = (
        np.longdouble(geometry.count)
        * np.longdouble(mean_weight[0])
    )

    scan_key = hash_items(
        VERSION,
        path.name,
        fingerprint,
        geometry.scale,
        geometry.characters,
        final_fit.parameters,
        next_input,
        SCAN_BLOCK,
    )

    scan_path = CACHE_DIR / (
        "scan_" + scan_key + ".npz"
    )

    say()
    say(
        "Trazim globalni maksimum zajednicke koherentne raspodele."
    )

    scan = exhaustive_next(
        geometry,
        next_amplitudes,
        checkpoint=scan_path,
        verbose=True,
    )

    normalization_ratio = float(
        scan["total_weight"] / analytic_total
    )

    normalization_error = abs(
        normalization_ratio - 1.0
    )

    require(
        normalization_error < 2.0e-8,
        "Potpuna pretraga nije potvrdila analiticku normalizaciju.",
    )

    analytic_inclusion = (
        geometry.inclusion_probabilities(
            next_amplitudes[None, :]
        )[0]
    )

    inclusion_error = float(
        np.max(
            np.abs(
                analytic_inclusion - scan["inclusion"]
            )
        )
    )

    require(
        inclusion_error < 2.0e-8,
        "Potpuna pretraga nije potvrdila marginalne verovatnoce.",
    )

    source_error, interference_error, physical_probability = (
        qiskit_interference_check(
            source,
            final_fit,
            next_input[0],
            geometry,
            scan["selected"],
            next_amplitudes,
        )
    )

    next_probability = float(
        np.longdouble(scan["weight"])
        / scan["total_weight"]
    )

    require(
        0.0 < next_probability <= 1.0
        and next_probability
        >= 1.0 / geometry.count - 1.0e-14,
        "Verovatnoca globalnog maksimuma nije ispravna.",
    )

    recent_inputs = all_inputs[-8:]

    recent_amplitudes = predict_amplitudes(
        source,
        final_fit,
        recent_inputs,
    )

    recent_inclusion = (
        geometry.inclusion_probabilities(
            recent_amplitudes
        )
    )

    sensitivity = float(
        np.max(np.ptp(recent_inclusion, axis=0))
    )

    numbers = tuple(
        int(value + 1)
        for value in scan["selected"]
    )

    warnings = []

    if maximum_run(numbers) >= 5:
        warnings.append(
            "NEXT sadrzi najmanje pet uzastopnih brojeva."
        )

    if numbers[-1] - numbers[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if any(
        np.array_equal(
            scan["selected"], historical
        )
        for historical in draws
    ):
        warnings.append(
            "NEXT ponavlja istorijsku kombinaciju."
        )

    if np.array_equal(
        scan["selected"], draws[-1]
    ):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Izlaz slabo reaguje na proverene istorijske kontekste."
        )

    if final_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Stacionarnost zavrsne optimizacije nije potvrdjena."
        )

    if test_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Test model ima nepotvrdjenu stacionarnost optimizacije."
        )

    say()
    say("REZULTAT:", path.name)
    say("NEXT:", " ".join(map(str, numbers)))
    say(f"Izabrana geometrijska skala: {scale:.2f}")

    say(
        f"Validacioni NLL: "
        f"{selected['validation']:.9f}"
    )

    say(f"Odvojeni test NLL: {test_nll:.9f}")

    say(
        f"Modelna verovatnoca NEXT: "
        f"{next_probability:.12g}"
    )

    say(
        f"Zbir svih verovatnoca: "
        f"{normalization_ratio:.12f}"
    )

    say(
        f"Pregledanih kombinacija: "
        f"{scan['visited']:,}"
    )

    say(
        f"Greska potpune normalizacije: "
        f"{normalization_error:.3e}"
    )

    say(
        f"Greska marginalnih verovatnoca: "
        f"{inclusion_error:.3e}"
    )

    say(
        f"Greska Qiskit izvora: "
        f"{source_error:.3e}"
    )

    say(
        f"Greska Qiskit interferencije: "
        f"{interference_error:.3e}"
    )

    say(
        f"Verovatnoca izlaza 0 u interferencijskom kolu: "
        f"{physical_probability:.12g}"
    )

    say(
        "Ova poslednja verovatnoca nije normalizovana "
        "verovatnoca Loto kombinacije."
    )

    say(
        f"Dijagnosticka osetljivost: "
        f"{sensitivity:.9e}"
    )

    say(
        f"Zavrsni projektovani gradijent: "
        f"{final_fit.gradient:.3e}"
    )

    say(
        f"Zbir verovatnoca ukljucivanja: "
        f"{scan['inclusion'].sum():.12f}"
    )

    say(
        "Verovatnoce ukljucivanja brojeva "
        "u zajednickoj NEXT raspodeli:"
    )

    for number, probability in enumerate(
        scan["inclusion"], start=1
    ):
        say(f"{number:2d}: {probability:.9f}")

    for warning in warnings:
        say("DIJAGNOSTIKA:", warning)

    if warnings:
        say(
            "Rezultat nije menjan radi uklanjanja "
            "dijagnostickog obrasca."
        )

    return {
        "name": path.name,
        "next": numbers,
        "scale": scale,
        "test_nll": test_nll,
        "inclusion": scan["inclusion"],
        "sensitivity": sensitivity,
        "last": tuple(
            int(value + 1)
            for value in draws[-1]
        ),
    }


def resource_report(loaded, source):
    maximum_rows = max(
        len(draws) for _, draws, _ in loaded
    )

    padded = next_power_of_two(maximum_rows)

    state_bytes = padded * MODES * 16

    four_count = math.comb(NUMBERS, 4)
    four_bytes = four_count * MODES * 16

    estimated = (
        384 * 1024**2
        + 12 * state_bytes
        + (len(GEOMETRY_SCALES) + 2) * four_bytes
        + 12 * maximum_rows * MODES * 16
    )

    circuit = source.circuit(source.seed())

    say()
    say("PROCENA RESURSA")
    say(f"Kubita talasnih modova: {source.qubits}")
    say(f"Talasnih komponenti: {source.modes}")
    say(f"Slojeva kvantnog izvora: {source.layers}")
    say(f"Naucenih uglova: {source.size}")
    say(f"Kapija naucenog izvora: {len(circuit.data)}")
    say(f"Dubina naucenog izvora: {circuit.depth()}")

    say(
        "Priprema ulaznih amplituda i zavrsno fazno kolo "
        "nisu ukljuceni u prethodni broj kapija."
    )

    say(
        f"Tabela cetvoroclanih podskupova po skali: "
        f"{four_bytes / 1024**2:.1f} MiB"
    )

    say(
        f"Konzervativna procena radne memorije: "
        f"{estimated / 1024**3:.2f} GiB"
    )

    say(
        "Ovo je procena radnih nizova, "
        "a ne izmereni vrh memorije procesa."
    )

    require(
        estimated <= 4.0 * 1024**3,
        "Procenjena memorija prelazi cetiri GiB. "
        "Trening nije zapocet.",
    )


def main():
    say("QUANTUM CAIRO V1 — Loto 7/39")
    say("Python:", sys.version.split()[0])
    say("Qiskit:", qiskit.__version__)
    say("NumPy:", np.__version__)
    say("SciPy:", scipy.__version__)
    say()

    say(
        "Geometrija suma podskupova i koherentna "
        "Fourierova interferencija."
    )

    say(
        "Kompleksni doprinosi cele kombinacije sabiraju se "
        "pre kvadriranja modula."
    )

    say(
        "Qiskit racuna naucene amplitude tokom "
        "treninga, validacije, testa i NEXT proracuna."
    )

    say(
        "Ponavljani interferencijski izlaz u treningu "
        "racuna se tacnom analitickom formulom."
    )

    say(
        "NEXT dodatno koristi eksplicitno Qiskit "
        "fazno kolo i Hadamardove kapije."
    )

    say(
        "Normalizacija, gradijent, optimizator i globalna "
        "pretraga su klasicni numericki proracuni."
    )

    say(
        "Nema klasicnog prediktivnog regresora "
        "koji zamenjuje kvantni izvor."
    )

    say(
        "Kvantna kola izvrsavaju se na lokalnom simulatoru."
    )

    say(
        "Nema nasumicnog generisanja "
        "ni uzorkovanja merenja."
    )

    say(
        "Momentna geometrija koristi svih sedam stepenih suma."
    )

    say(
        "Geometrijska jedinstvenost ne zahteva "
        "razlicitu verovatnocu svake kombinacije."
    )

    say(
        "Izbor skale koristi hronolosku validaciju; "
        "test ostaje odvojen."
    )

    say(
        "Test je jedan korak unapred: prethodno stvarno "
        "kolo dostupno je kao istorijski ulaz."
    )

    say(
        "Zavrsni trening koristi sve istorijske prelaze."
    )

    say(
        f"Budzet po treningu: {MAX_ITERATIONS} iteracija "
        f"i {MAX_EVALUATIONS} optimizacionih evaluacija."
    )

    say(
        "Pocetna i zavrsna provera funkcije cilja "
        "rade se dodatno."
    )

    say(
        "Lokalna optimizacija ne garantuje globalni optimum parametara."
    )

    say(
        "Za dobijene parametre NEXT maksimum se "
        "trazi pregledom svih kombinacija."
    )

    loaded = []

    for path in FILES:
        draws, fingerprint = load_draws(path)
        loaded.append(
            (path, draws, fingerprint)
        )

    source = QuantumSource()

    resource_report(loaded, source)
    self_checks()

    geometries = {}

    for scale in GEOMETRY_SCALES:
        geometry = Geometry(scale)
        geometries[scale] = geometry

        say()
        say(f"Geometrijska skala: {scale:.2f}")

        say(
            f"Najmanja sopstvena vrednost Gramove matrice: "
            f"{geometry.minimum_eigenvalue:.9e}"
        )

        say(
            f"Najmanje razdvajanje pojedinacne zamene: "
            f"{geometry.minimum_single_swap_separation:.9e}"
        )

    CACHE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    say()
    say("Kontrolne tacke:", str(CACHE_DIR))

    results = []

    for path, draws, fingerprint in loaded:
        result = process_file(
            path,
            draws,
            fingerprint,
            source,
            geometries,
        )

        results.append(result)

    say()
    say("=" * 70)
    say("ZAVRSNI NEXT REZULTATI")

    for result in results:
        say(
            f"{result['name']} | NEXT: "
            + " ".join(map(str, result["next"]))
            + f" | skala: {result['scale']:.2f}"
            + f" | test NLL: {result['test_nll']:.9f}"
        )

    for first, second in combinations(results, 2):
        if first["next"] == second["next"]:
            difference = float(
                np.max(
                    np.abs(
                        first["inclusion"]
                        - second["inclusion"]
                    )
                )
            )

            say()
            say(
                "DIJAGNOSTIKA: isti NEXT za",
                first["name"],
                "i",
                second["name"],
            )

            say(
                f"Najveca razlika izlaznih verovatnoca: "
                f"{difference:.9e}"
            )

            say(
                f"Osetljivosti na istoriju: "
                f"{first['sensitivity']:.9e} i "
                f"{second['sensitivity']:.9e}"
            )

            say(
                "Isto poslednje istorijsko kolo:",
                first["last"] == second["last"],
            )

            say(
                "Rezultat nije menjan "
                "radi vestackog razlikovanja."
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say()
        say("PREKINUTO NA ZAHTEV KORISNIKA.")

        say(
            "Trening cuva najbolji kandidat. "
            "Pretraga se nastavlja od poslednjeg "
            "potpuno sacuvanog prefiksa."
        )

        sys.exit(130)

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say()
        say("ZAUSTAVLJENO:", str(error))

        say(
            "Zavrsni skup od tri NEXT rezultata "
            "nije potvrdjen."
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM CAIRO V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
NumPy: 1.26.4
SciPy: 1.17.1

Geometrija suma podskupova i koherentna Fourierova interferencija.
Kompleksni doprinosi cele kombinacije sabiraju se pre kvadriranja modula.
Qiskit racuna naucene amplitude tokom treninga, validacije, testa i NEXT proracuna.
Ponavljani interferencijski izlaz u treningu racuna se tacnom analitickom formulom.
NEXT dodatno koristi eksplicitno Qiskit fazno kolo i Hadamardove kapije.
Normalizacija, gradijent, optimizator i globalna pretraga su klasicni numericki proracuni.
Nema klasicnog prediktivnog regresora koji zamenjuje kvantni izvor.
Kvantna kola izvrsavaju se na lokalnom simulatoru.
Nema nasumicnog generisanja ni uzorkovanja merenja.
Momentna geometrija koristi svih sedam stepenih suma.
Geometrijska jedinstvenost ne zahteva razlicitu verovatnocu svake kombinacije.
Izbor skale koristi hronolosku validaciju; test ostaje odvojen.
Test je jedan korak unapred: prethodno stvarno kolo dostupno je kao istorijski ulaz.
Zavrsni trening koristi sve istorijske prelaze.
Budzet po treningu: 80 iteracija i 120 optimizacionih evaluacija.
Pocetna i zavrsna provera funkcije cilja rade se dodatno.
Lokalna optimizacija ne garantuje globalni optimum parametara.
Za dobijene parametre NEXT maksimum se trazi pregledom svih kombinacija.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

PROCENA RESURSA
Kubita talasnih modova: 6
Talasnih komponenti: 64
Slojeva kvantnog izvora: 3
Naucenih uglova: 54
Kapija naucenog izvora: 54
Dubina naucenog izvora: 24
Priprema ulaznih amplituda i zavrsno fazno kolo nisu ukljuceni u prethodni broj kapija.
Tabela cetvoroclanih podskupova po skali: 80.3 MiB
Konzervativna procena radne memorije: 0.84 GiB
Ovo je procena radnih nizova, a ne izmereni vrh memorije procesa.

Pokrecem matematicke i Qiskit samoprovere.
Samoprovere su prosle.
Najveca greska gradijenta: 3.149e-10
Najveca greska paketne simulacije: 4.460e-16
Greska eksplicitne interferencije: 2.082e-17

Geometrijska skala: 0.75
Najmanja sopstvena vrednost Gramove matrice: 3.823764892e-02
Najmanje razdvajanje pojedinacne zamene: 7.197205541e-03

Geometrijska skala: 1.50
Najmanja sopstvena vrednost Gramove matrice: 1.493944793e-01
Najmanje razdvajanje pojedinacne zamene: 2.853610946e-02

Kontrolne tacke: /Quantum_Cairo/Quantum_Cairo_V1_checkpoints

======================================================================
OBRADA: loto7_4710_k86.csv
Istorijskih prelaza: 4709
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86.csv | skala 0.75 | validacija 1
Skala: 0.75 | uzorci: 2825 | parametri: 54
Evaluacija 1/120 | cilj 17.476749488 | gradijent 1.210e-01 | proteklo 0.5 s
Evaluacija 5/120 | cilj 17.084466799 | gradijent 9.573e-02 | proteklo 1.4 s
Evaluacija 10/120 | cilj 17.057737393 | gradijent 1.682e-01 | proteklo 2.6 s
Evaluacija 15/120 | cilj 17.047029919 | gradijent 6.669e-02 | proteklo 3.7 s
Evaluacija 20/120 | cilj 17.010263258 | gradijent 1.972e-01 | proteklo 4.8 s
Evaluacija 25/120 | cilj 16.981926749 | gradijent 4.126e-02 | proteklo 5.9 s
Evaluacija 30/120 | cilj 16.960195807 | gradijent 4.200e-02 | proteklo 7.0 s
Evaluacija 35/120 | cilj 16.948431730 | gradijent 2.269e-02 | proteklo 8.1 s
Evaluacija 40/120 | cilj 16.934880430 | gradijent 4.131e-02 | proteklo 9.2 s
Evaluacija 45/120 | cilj 16.921964766 | gradijent 2.227e-02 | proteklo 10.3 s
Evaluacija 50/120 | cilj 16.909293640 | gradijent 2.648e-02 | proteklo 11.4 s
Evaluacija 55/120 | cilj 16.898310240 | gradijent 1.883e-02 | proteklo 12.5 s
Evaluacija 60/120 | cilj 16.890432668 | gradijent 1.601e-02 | proteklo 13.6 s
Evaluacija 65/120 | cilj 16.884757012 | gradijent 9.799e-03 | proteklo 14.8 s
Evaluacija 70/120 | cilj 16.880637809 | gradijent 1.166e-02 | proteklo 15.9 s
Evaluacija 75/120 | cilj 16.873930341 | gradijent 1.971e-02 | proteklo 17.0 s
Evaluacija 80/120 | cilj 16.865861288 | gradijent 1.337e-02 | proteklo 18.1 s
Evaluacija 85/120 | cilj 16.857913196 | gradijent 3.609e-02 | proteklo 19.1 s
Evaluacija 90/120 | cilj 16.845685656 | gradijent 1.563e-02 | proteklo 20.2 s
Evaluacija 95/120 | cilj 16.837638238 | gradijent 1.213e-02 | proteklo 21.4 s
Najbolji cilj: 16.835276874
Max|projektovani gradijent|: 1.230e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.121922793

loto7_4710_k86.csv | skala 0.75 | validacija 2
Skala: 0.75 | uzorci: 3296 | parametri: 54
Evaluacija 1/120 | cilj 16.876981727 | gradijent 3.622e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.867017047 | gradijent 8.925e-03 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.862105950 | gradijent 1.178e-02 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.858354828 | gradijent 1.290e-02 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.855572430 | gradijent 4.638e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.853310254 | gradijent 6.908e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.851202684 | gradijent 8.960e-03 | proteklo 6.9 s
Evaluacija 35/120 | cilj 16.850339100 | gradijent 3.906e-03 | proteklo 8.0 s
Evaluacija 40/120 | cilj 16.850014874 | gradijent 2.242e-03 | proteklo 9.1 s
Evaluacija 45/120 | cilj 16.849941548 | gradijent 5.654e-04 | proteklo 10.2 s
Evaluacija 50/120 | cilj 16.849918704 | gradijent 3.783e-04 | proteklo 11.3 s
Evaluacija 55/120 | cilj 16.849913216 | gradijent 1.442e-04 | proteklo 12.6 s
Evaluacija 60/120 | cilj 16.849912129 | gradijent 1.844e-04 | proteklo 13.7 s
Evaluacija 65/120 | cilj 16.849907059 | gradijent 9.250e-04 | proteklo 14.8 s
Evaluacija 70/120 | cilj 16.849866937 | gradijent 1.988e-03 | proteklo 15.9 s
Evaluacija 75/120 | cilj 16.849672538 | gradijent 3.972e-03 | proteklo 17.0 s
Evaluacija 80/120 | cilj 16.849241244 | gradijent 2.736e-03 | proteklo 18.1 s
Evaluacija 85/120 | cilj 16.848859926 | gradijent 6.679e-03 | proteklo 19.2 s
Najbolji cilj: 16.848703262
Max|projektovani gradijent|: 2.823e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.111782406

Skala 0.75 | objedinjeni validacioni NLL: 17.116852600

loto7_4710_k86.csv | skala 1.50 | validacija 1
Skala: 1.50 | uzorci: 2825 | parametri: 54
Evaluacija 1/120 | cilj 17.265109484 | gradijent 1.126e-01 | proteklo 0.4 s
Evaluacija 5/120 | cilj 17.084865144 | gradijent 9.706e-02 | proteklo 1.3 s
Evaluacija 10/120 | cilj 17.065139553 | gradijent 6.250e-02 | proteklo 2.4 s
Evaluacija 15/120 | cilj 17.045453576 | gradijent 1.356e-01 | proteklo 3.5 s
Evaluacija 20/120 | cilj 17.041720155 | gradijent 5.106e-02 | proteklo 4.6 s
Evaluacija 25/120 | cilj 17.032100893 | gradijent 1.235e-01 | proteklo 5.7 s
Evaluacija 30/120 | cilj 17.002064170 | gradijent 5.121e-02 | proteklo 6.8 s
Evaluacija 35/120 | cilj 16.986171150 | gradijent 2.846e-02 | proteklo 7.9 s
Evaluacija 40/120 | cilj 16.980302977 | gradijent 1.420e-02 | proteklo 9.1 s
Evaluacija 45/120 | cilj 16.976644623 | gradijent 1.155e-02 | proteklo 10.2 s
Evaluacija 50/120 | cilj 16.973801462 | gradijent 1.267e-02 | proteklo 11.3 s
Evaluacija 55/120 | cilj 16.968540564 | gradijent 1.294e-02 | proteklo 12.4 s
Evaluacija 60/120 | cilj 16.963363895 | gradijent 9.633e-03 | proteklo 13.5 s
Evaluacija 65/120 | cilj 16.959914268 | gradijent 2.763e-02 | proteklo 14.6 s
Evaluacija 70/120 | cilj 16.953524052 | gradijent 3.234e-02 | proteklo 15.7 s
Evaluacija 75/120 | cilj 16.943501001 | gradijent 3.457e-02 | proteklo 16.8 s
Evaluacija 80/120 | cilj 16.933175996 | gradijent 5.626e-02 | proteklo 17.9 s
Evaluacija 85/120 | cilj 16.921378744 | gradijent 3.236e-02 | proteklo 19.0 s
Evaluacija 90/120 | cilj 16.913236892 | gradijent 1.589e-02 | proteklo 20.1 s
Evaluacija 95/120 | cilj 16.909822749 | gradijent 1.235e-02 | proteklo 21.2 s
Najbolji cilj: 16.906955113
Max|projektovani gradijent|: 1.029e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.104670980

loto7_4710_k86.csv | skala 1.50 | validacija 2
Skala: 1.50 | uzorci: 3296 | parametri: 54
Evaluacija 1/120 | cilj 16.935472460 | gradijent 2.432e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.930207305 | gradijent 7.914e-03 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.926023093 | gradijent 4.511e-03 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.924947894 | gradijent 3.567e-03 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.923396607 | gradijent 7.761e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.921585650 | gradijent 3.066e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.919320531 | gradijent 8.759e-03 | proteklo 6.8 s
Evaluacija 35/120 | cilj 16.915373154 | gradijent 1.171e-02 | proteklo 7.9 s
Evaluacija 40/120 | cilj 16.912312895 | gradijent 9.107e-03 | proteklo 9.0 s
Evaluacija 45/120 | cilj 16.909189209 | gradijent 8.060e-03 | proteklo 10.2 s
Evaluacija 50/120 | cilj 16.906843616 | gradijent 5.233e-03 | proteklo 11.4 s
Evaluacija 55/120 | cilj 16.905242795 | gradijent 8.467e-03 | proteklo 12.6 s
Evaluacija 60/120 | cilj 16.903205053 | gradijent 8.933e-03 | proteklo 13.7 s
Evaluacija 65/120 | cilj 16.901130150 | gradijent 1.782e-02 | proteklo 14.8 s
Evaluacija 70/120 | cilj 16.897214969 | gradijent 3.396e-03 | proteklo 15.9 s
Evaluacija 75/120 | cilj 16.894215529 | gradijent 8.269e-03 | proteklo 17.0 s
Evaluacija 80/120 | cilj 16.891815953 | gradijent 7.650e-03 | proteklo 18.1 s
Evaluacija 85/120 | cilj 16.890359345 | gradijent 7.147e-03 | proteklo 19.2 s
Najbolji cilj: 16.890003459
Max|projektovani gradijent|: 5.815e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.022243921

Skala 1.50 | objedinjeni validacioni NLL: 17.063457451

IZBOR GEOMETRIJSKE SKALE
Skala 0.75 | validacioni NLL 17.116852600
Skala 1.50 | validacioni NLL 17.063457451
Izabrana skala: 1.50
Tolerancija pri izboru manje skale: 0.000010 NLL

loto7_4710_k86.csv | odvojeni test
Skala: 1.50 | uzorci: 3767 | parametri: 54
Evaluacija 1/120 | cilj 16.907100437 | gradijent 3.885e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.901153904 | gradijent 1.021e-02 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.899624288 | gradijent 3.177e-03 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.899335807 | gradijent 2.403e-03 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.899057673 | gradijent 4.000e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.898477884 | gradijent 5.670e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.896785090 | gradijent 5.722e-03 | proteklo 6.8 s
Evaluacija 35/120 | cilj 16.895347517 | gradijent 2.448e-03 | proteklo 8.0 s
Evaluacija 40/120 | cilj 16.894464064 | gradijent 2.620e-03 | proteklo 9.2 s
Evaluacija 45/120 | cilj 16.894227623 | gradijent 1.177e-03 | proteklo 10.3 s
Evaluacija 50/120 | cilj 16.894181198 | gradijent 5.309e-04 | proteklo 11.4 s
Evaluacija 55/120 | cilj 16.894171337 | gradijent 2.235e-04 | proteklo 12.5 s
Evaluacija 60/120 | cilj 16.894170034 | gradijent 1.277e-04 | proteklo 13.6 s
Evaluacija 65/120 | cilj 16.894169723 | gradijent 1.460e-04 | proteklo 14.7 s
Evaluacija 70/120 | cilj 16.894169466 | gradijent 5.251e-05 | proteklo 15.8 s
Evaluacija 75/120 | cilj 16.894169385 | gradijent 2.952e-05 | proteklo 16.9 s
Evaluacija 80/120 | cilj 16.894169319 | gradijent 1.860e-05 | proteklo 18.0 s
Najbolji cilj: 16.894169302
Max|projektovani gradijent|: 1.981e-05
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Odvojeni test NLL: 17.049586846
Test ne menja izabranu skalu.

loto7_4710_k86.csv | zavrsni trening cele istorije
Skala: 1.50 | uzorci: 4709 | parametri: 54
Evaluacija 1/120 | cilj 16.926199264 | gradijent 2.764e-02 | proteklo 0.9 s
Evaluacija 5/120 | cilj 16.919889685 | gradijent 1.735e-02 | proteklo 2.8 s
Evaluacija 10/120 | cilj 16.917718266 | gradijent 5.724e-03 | proteklo 5.1 s
Evaluacija 15/120 | cilj 16.915258050 | gradijent 5.758e-03 | proteklo 7.4 s
Evaluacija 20/120 | cilj 16.913513246 | gradijent 9.410e-03 | proteklo 9.6 s
Evaluacija 25/120 | cilj 16.910746881 | gradijent 4.847e-03 | proteklo 11.9 s
Evaluacija 30/120 | cilj 16.909937746 | gradijent 2.348e-03 | proteklo 14.2 s
Evaluacija 35/120 | cilj 16.909203210 | gradijent 3.766e-03 | proteklo 16.5 s
Evaluacija 40/120 | cilj 16.908936855 | gradijent 2.157e-03 | proteklo 18.8 s
Evaluacija 45/120 | cilj 16.908717205 | gradijent 2.484e-03 | proteklo 21.1 s
Evaluacija 50/120 | cilj 16.908607965 | gradijent 1.301e-03 | proteklo 23.4 s
Evaluacija 55/120 | cilj 16.908395534 | gradijent 3.232e-03 | proteklo 25.7 s
Evaluacija 60/120 | cilj 16.908067923 | gradijent 2.828e-03 | proteklo 28.0 s
Evaluacija 65/120 | cilj 16.908000671 | gradijent 2.741e-03 | proteklo 30.3 s
Evaluacija 70/120 | cilj 16.907912883 | gradijent 2.790e-03 | proteklo 32.6 s
Evaluacija 75/120 | cilj 16.907784553 | gradijent 2.306e-03 | proteklo 34.9 s
Evaluacija 80/120 | cilj 16.907594609 | gradijent 1.406e-03 | proteklo 37.2 s
Najbolji cilj: 16.907532170
Max|projektovani gradijent|: 1.590e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.

Trazim globalni maksimum zajednicke koherentne raspodele.
Pregledano 2,022,398/15,380,937 | 13.15% | proteklo 0.1 s
Pregledano 2,685,276/15,380,937 | 17.46% | proteklo 0.2 s
Pregledano 3,364,025/15,380,937 | 21.87% | proteklo 0.2 s
Pregledano 4,702,537/15,380,937 | 30.57% | proteklo 0.3 s
Pregledano 5,070,086/15,380,937 | 32.96% | proteklo 0.3 s
Pregledano 6,126,603/15,380,937 | 39.83% | proteklo 0.4 s
Pregledano 6,923,742/15,380,937 | 45.02% | proteklo 0.5 s
Pregledano 7,299,885/15,380,937 | 47.46% | proteklo 0.5 s
Pregledano 8,392,538/15,380,937 | 54.56% | proteklo 0.6 s
Pregledano 8,653,288/15,380,937 | 56.26% | proteklo 0.6 s
Pregledano 9,623,294/15,380,937 | 62.57% | proteklo 0.7 s
Pregledano 9,988,483/15,380,937 | 64.94% | proteklo 0.7 s
Pregledano 10,720,521/15,380,937 | 69.70% | proteklo 0.8 s
Pregledano 11,090,325/15,380,937 | 72.10% | proteklo 0.8 s
Pregledano 11,703,567/15,380,937 | 76.09% | proteklo 0.8 s
Pregledano 12,007,052/15,380,937 | 78.06% | proteklo 0.9 s
Pregledano 12,572,260/15,380,937 | 81.74% | proteklo 0.9 s
Pregledano 12,751,026/15,380,937 | 82.90% | proteklo 0.9 s
Pregledano 13,289,586/15,380,937 | 86.40% | proteklo 1.0 s
Pregledano 13,581,359/15,380,937 | 88.30% | proteklo 1.0 s
Pregledano 13,817,154/15,380,937 | 89.83% | proteklo 1.0 s
Pregledano 14,158,136/15,380,937 | 92.05% | proteklo 1.1 s
Pregledano 14,391,939/15,380,937 | 93.57% | proteklo 1.1 s
Pregledano 14,576,913/15,380,937 | 94.77% | proteklo 1.1 s
Pregledano 14,723,124/15,380,937 | 95.72% | proteklo 1.1 s
Pregledano 14,900,027/15,380,937 | 96.87% | proteklo 1.1 s
Pregledano 15,034,820/15,380,937 | 97.75% | proteklo 1.2 s
Pregledano 15,170,346/15,380,937 | 98.63% | proteklo 1.2 s
Pregledano 15,257,650/15,380,937 | 99.20% | proteklo 1.2 s
Pregledano 15,313,805/15,380,937 | 99.56% | proteklo 1.2 s
Pregledano 15,356,428/15,380,937 | 99.84% | proteklo 1.2 s
Pregledano 15,376,987/15,380,937 | 99.97% | proteklo 1.2 s
Pregledano 15,380,937/15,380,937 | 100.00% | proteklo 1.2 s

REZULTAT: loto7_4710_k86.csv
NEXT: 2 x 22 y 30 z 36
Izabrana geometrijska skala: 1.50
Validacioni NLL: 17.063457451
Odvojeni test NLL: 17.049586846
Modelna verovatnoca NEXT: 5.79601665391e-07
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska potpune normalizacije: 2.442e-15
Greska marginalnih verovatnoca: 1.471e-15
Greska Qiskit izvora: 0.000e+00
Greska Qiskit interferencije: 1.665e-16
Verovatnoca izlaza 0 u interferencijskom kolu: 0.137900655008
Ova poslednja verovatnoca nije normalizovana verovatnoca Loto kombinacije.
Dijagnosticka osetljivost: 2.333732011e-02
Zavrsni projektovani gradijent: 1.590e-03
Zbir verovatnoca ukljucivanja: 7.000000000000
Verovatnoce ukljucivanja brojeva u zajednickoj NEXT raspodeli:
 1: 0.177273620
 2: 0.176141746
 3: 0.175245715
 4: 0.174578121
 5: 0.174118170
 6: 0.173836816
 7: 0.173773266
 8: 0.173971216
 9: 0.174331556
10: 0.174709788
11: 0.175077022
12: 0.175484470
13: 0.175968903
14: 0.176544406
15: 0.177227831
16: 0.178006638
17: 0.178784135
18: 0.179440037
19: 0.179844056
20: 0.180154282
21: 0.180543976
22: 0.180917740
23: 0.181496417
24: 0.181609979
25: 0.180818416
26: 0.180451103
27: 0.181325942
28: 0.185627319
29: 0.186642281
30: 0.184787837
31: 0.182644506
32: 0.190997395
33: 0.188498269
34: 0.183694230
35: 0.178011285
36: 0.177467295
37: 0.188422223
38: 0.181597182
39: 0.179934810
DIJAGNOSTIKA: Stacionarnost zavrsne optimizacije nije potvrdjena.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4710_k86_loto_2977.csv
Istorijskih prelaza: 2976
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86_loto_2977.csv | skala 0.75 | validacija 1
Skala: 0.75 | uzorci: 1785 | parametri: 54
Evaluacija 1/120 | cilj 17.482251864 | gradijent 1.612e-01 | proteklo 0.2 s
Evaluacija 5/120 | cilj 17.040601685 | gradijent 7.783e-02 | proteklo 0.7 s
Evaluacija 10/120 | cilj 17.010887118 | gradijent 8.902e-02 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.960961010 | gradijent 8.126e-02 | proteklo 1.8 s
Evaluacija 20/120 | cilj 16.926268823 | gradijent 5.082e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.905213546 | gradijent 2.637e-02 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.891580409 | gradijent 2.804e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.871778920 | gradijent 2.545e-02 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.857378104 | gradijent 2.130e-02 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.847698178 | gradijent 1.842e-02 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.838286382 | gradijent 2.034e-02 | proteklo 5.5 s
Evaluacija 55/120 | cilj 16.831876811 | gradijent 1.420e-02 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.825565396 | gradijent 1.137e-02 | proteklo 6.6 s
Evaluacija 65/120 | cilj 16.819909010 | gradijent 1.302e-02 | proteklo 7.2 s
Evaluacija 70/120 | cilj 16.811734496 | gradijent 1.035e-02 | proteklo 7.7 s
Evaluacija 75/120 | cilj 16.806276234 | gradijent 7.811e-03 | proteklo 8.3 s
Evaluacija 80/120 | cilj 16.802630264 | gradijent 8.503e-03 | proteklo 8.8 s
Evaluacija 85/120 | cilj 16.798909847 | gradijent 1.353e-02 | proteklo 9.4 s
Najbolji cilj: 16.795997580
Max|projektovani gradijent|: 5.881e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.066387077

loto7_4710_k86_loto_2977.csv | skala 0.75 | validacija 2
Skala: 0.75 | uzorci: 2083 | parametri: 54
Evaluacija 1/120 | cilj 16.835147635 | gradijent 3.088e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.827797671 | gradijent 7.180e-03 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.824346800 | gradijent 8.552e-03 | proteklo 2.5 s
Evaluacija 15/120 | cilj 16.823118701 | gradijent 3.467e-03 | proteklo 3.7 s
Evaluacija 20/120 | cilj 16.822207568 | gradijent 4.103e-03 | proteklo 4.8 s
Evaluacija 25/120 | cilj 16.820802041 | gradijent 4.478e-03 | proteklo 5.9 s
Evaluacija 30/120 | cilj 16.817142220 | gradijent 1.161e-02 | proteklo 7.0 s
Evaluacija 35/120 | cilj 16.814093465 | gradijent 1.121e-02 | proteklo 8.1 s
Evaluacija 40/120 | cilj 16.812029676 | gradijent 6.538e-03 | proteklo 9.2 s
Evaluacija 45/120 | cilj 16.809987663 | gradijent 4.679e-03 | proteklo 10.3 s
Evaluacija 50/120 | cilj 16.808661078 | gradijent 9.166e-03 | proteklo 11.4 s
Evaluacija 55/120 | cilj 16.807887691 | gradijent 5.939e-03 | proteklo 12.5 s
Evaluacija 60/120 | cilj 16.806402789 | gradijent 6.136e-03 | proteklo 13.5 s
Evaluacija 65/120 | cilj 16.805014676 | gradijent 5.629e-03 | proteklo 14.6 s
Evaluacija 70/120 | cilj 16.804245346 | gradijent 4.627e-03 | proteklo 15.7 s
Evaluacija 75/120 | cilj 16.803439597 | gradijent 2.566e-03 | proteklo 16.8 s
Evaluacija 80/120 | cilj 16.803196125 | gradijent 2.870e-03 | proteklo 17.9 s
Najbolji cilj: 16.803127856
Max|projektovani gradijent|: 1.152e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.128174859

Skala 0.75 | objedinjeni validacioni NLL: 17.097229046

loto7_4710_k86_loto_2977.csv | skala 1.50 | validacija 1
Skala: 1.50 | uzorci: 1785 | parametri: 54
Evaluacija 1/120 | cilj 17.247961668 | gradijent 1.180e-01 | proteklo 0.2 s
Evaluacija 5/120 | cilj 17.022715201 | gradijent 9.055e-02 | proteklo 0.7 s
Evaluacija 10/120 | cilj 16.977370278 | gradijent 7.134e-02 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.941214032 | gradijent 2.331e-02 | proteklo 1.8 s
Evaluacija 20/120 | cilj 16.927985479 | gradijent 1.814e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.921870920 | gradijent 1.334e-02 | proteklo 2.9 s
Evaluacija 30/120 | cilj 16.911231208 | gradijent 1.336e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.895577358 | gradijent 2.895e-02 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.883186827 | gradijent 5.353e-02 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.873636358 | gradijent 2.707e-02 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.858389985 | gradijent 1.515e-02 | proteklo 5.5 s
Evaluacija 55/120 | cilj 16.845091171 | gradijent 1.671e-02 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.840226203 | gradijent 9.646e-03 | proteklo 6.6 s
Evaluacija 65/120 | cilj 16.836852161 | gradijent 1.412e-02 | proteklo 7.2 s
Evaluacija 70/120 | cilj 16.832197936 | gradijent 8.323e-03 | proteklo 7.7 s
Evaluacija 75/120 | cilj 16.826177690 | gradijent 2.114e-02 | proteklo 8.3 s
Evaluacija 80/120 | cilj 16.813447783 | gradijent 2.890e-02 | proteklo 8.8 s
Najbolji cilj: 16.801778071
Max|projektovani gradijent|: 1.281e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 16.950274945

loto7_4710_k86_loto_2977.csv | skala 1.50 | validacija 2
Skala: 1.50 | uzorci: 2083 | parametri: 54
Evaluacija 1/120 | cilj 16.823519013 | gradijent 2.626e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.814996158 | gradijent 1.016e-02 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.811611038 | gradijent 5.816e-03 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.808691166 | gradijent 6.737e-03 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.806224478 | gradijent 4.695e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.805179541 | gradijent 2.967e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.804674932 | gradijent 2.116e-03 | proteklo 6.7 s
Evaluacija 35/120 | cilj 16.803990404 | gradijent 9.757e-03 | proteklo 7.9 s
Evaluacija 40/120 | cilj 16.802842416 | gradijent 1.772e-03 | proteklo 8.9 s
Evaluacija 45/120 | cilj 16.802674712 | gradijent 7.358e-04 | proteklo 10.0 s
Evaluacija 50/120 | cilj 16.802581982 | gradijent 3.127e-03 | proteklo 11.1 s
Evaluacija 55/120 | cilj 16.802413152 | gradijent 4.796e-03 | proteklo 12.2 s
Evaluacija 60/120 | cilj 16.801246887 | gradijent 3.054e-03 | proteklo 13.3 s
Evaluacija 65/120 | cilj 16.800907432 | gradijent 1.223e-03 | proteklo 14.4 s
Evaluacija 70/120 | cilj 16.800640410 | gradijent 1.668e-03 | proteklo 15.6 s
Evaluacija 75/120 | cilj 16.800370203 | gradijent 2.918e-03 | proteklo 16.7 s
Evaluacija 80/120 | cilj 16.799657698 | gradijent 8.194e-03 | proteklo 17.8 s
Evaluacija 85/120 | cilj 16.798856782 | gradijent 3.015e-03 | proteklo 18.8 s
Najbolji cilj: 16.798623316
Max|projektovani gradijent|: 2.216e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.083028346

Skala 1.50 | objedinjeni validacioni NLL: 17.016540088

IZBOR GEOMETRIJSKE SKALE
Skala 0.75 | validacioni NLL 17.097229046
Skala 1.50 | validacioni NLL 17.016540088
Izabrana skala: 1.50
Tolerancija pri izboru manje skale: 0.000010 NLL

loto7_4710_k86_loto_2977.csv | odvojeni test
Skala: 1.50 | uzorci: 2380 | parametri: 54
Evaluacija 1/120 | cilj 16.834740763 | gradijent 2.760e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.828561917 | gradijent 1.019e-02 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.826744074 | gradijent 3.325e-03 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.826440507 | gradijent 2.268e-03 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.826232043 | gradijent 3.229e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.825776355 | gradijent 2.314e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.825487600 | gradijent 2.786e-03 | proteklo 6.8 s
Evaluacija 35/120 | cilj 16.824858822 | gradijent 4.603e-03 | proteklo 7.9 s
Evaluacija 40/120 | cilj 16.823948315 | gradijent 4.112e-03 | proteklo 9.0 s
Evaluacija 45/120 | cilj 16.822718582 | gradijent 1.026e-02 | proteklo 10.1 s
Evaluacija 50/120 | cilj 16.819148794 | gradijent 1.098e-02 | proteklo 11.2 s
Evaluacija 55/120 | cilj 16.817165860 | gradijent 2.527e-03 | proteklo 12.3 s
Evaluacija 60/120 | cilj 16.816659192 | gradijent 1.823e-03 | proteklo 13.4 s
Evaluacija 65/120 | cilj 16.816414615 | gradijent 2.049e-03 | proteklo 14.5 s
Evaluacija 70/120 | cilj 16.816088185 | gradijent 1.955e-03 | proteklo 15.6 s
Evaluacija 75/120 | cilj 16.815736120 | gradijent 5.332e-03 | proteklo 16.7 s
Evaluacija 80/120 | cilj 16.814651064 | gradijent 7.725e-03 | proteklo 17.8 s
Najbolji cilj: 16.813303684
Max|projektovani gradijent|: 9.184e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.

Odvojeni test NLL: 17.134590201
Test ne menja izabranu skalu.

loto7_4710_k86_loto_2977.csv | zavrsni trening cele istorije
Skala: 1.50 | uzorci: 2976 | parametri: 54
Evaluacija 1/120 | cilj 16.878897346 | gradijent 4.268e-02 | proteklo 0.4 s
Evaluacija 5/120 | cilj 16.870016098 | gradijent 1.247e-02 | proteklo 1.3 s
Evaluacija 10/120 | cilj 16.860141312 | gradijent 1.021e-02 | proteklo 2.4 s
Evaluacija 15/120 | cilj 16.854833782 | gradijent 1.229e-02 | proteklo 3.5 s
Evaluacija 20/120 | cilj 16.853082098 | gradijent 2.505e-03 | proteklo 4.6 s
Evaluacija 25/120 | cilj 16.852650141 | gradijent 1.603e-03 | proteklo 5.7 s
Evaluacija 30/120 | cilj 16.852346023 | gradijent 7.171e-03 | proteklo 6.8 s
Evaluacija 35/120 | cilj 16.851287160 | gradijent 1.035e-02 | proteklo 7.9 s
Evaluacija 40/120 | cilj 16.849787632 | gradijent 8.441e-03 | proteklo 9.0 s
Evaluacija 45/120 | cilj 16.848058233 | gradijent 7.091e-03 | proteklo 10.1 s
Evaluacija 50/120 | cilj 16.846583076 | gradijent 2.639e-03 | proteklo 11.2 s
Evaluacija 55/120 | cilj 16.846010672 | gradijent 1.574e-03 | proteklo 12.3 s
Evaluacija 60/120 | cilj 16.845882513 | gradijent 1.067e-03 | proteklo 13.4 s
Evaluacija 65/120 | cilj 16.845702072 | gradijent 2.284e-03 | proteklo 14.6 s
Evaluacija 70/120 | cilj 16.845633642 | gradijent 7.261e-04 | proteklo 15.7 s
Evaluacija 75/120 | cilj 16.845621855 | gradijent 1.743e-04 | proteklo 16.9 s
Evaluacija 80/120 | cilj 16.845620020 | gradijent 6.671e-05 | proteklo 18.1 s
Evaluacija 85/120 | cilj 16.845619794 | gradijent 8.869e-05 | proteklo 19.2 s
Najbolji cilj: 16.845619794
Max|projektovani gradijent|: 8.869e-05
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.

Trazim globalni maksimum zajednicke koherentne raspodele.
Pregledano 2,022,398/15,380,937 | 13.15% | proteklo 0.1 s
Pregledano 2,685,276/15,380,937 | 17.46% | proteklo 0.2 s
Pregledano 3,364,025/15,380,937 | 21.87% | proteklo 0.2 s
Pregledano 4,702,537/15,380,937 | 30.57% | proteklo 0.3 s
Pregledano 5,070,086/15,380,937 | 32.96% | proteklo 0.3 s
Pregledano 6,126,603/15,380,937 | 39.83% | proteklo 0.4 s
Pregledano 6,923,742/15,380,937 | 45.02% | proteklo 0.5 s
Pregledano 7,299,885/15,380,937 | 47.46% | proteklo 0.5 s
Pregledano 8,392,538/15,380,937 | 54.56% | proteklo 0.6 s
Pregledano 8,653,288/15,380,937 | 56.26% | proteklo 0.6 s
Pregledano 9,623,294/15,380,937 | 62.57% | proteklo 0.7 s
Pregledano 9,988,483/15,380,937 | 64.94% | proteklo 0.7 s
Pregledano 10,720,521/15,380,937 | 69.70% | proteklo 0.7 s
Pregledano 11,090,325/15,380,937 | 72.10% | proteklo 0.8 s
Pregledano 11,703,567/15,380,937 | 76.09% | proteklo 0.8 s
Pregledano 12,007,052/15,380,937 | 78.06% | proteklo 0.8 s
Pregledano 12,572,260/15,380,937 | 81.74% | proteklo 0.9 s
Pregledano 12,751,026/15,380,937 | 82.90% | proteklo 0.9 s
Pregledano 13,289,586/15,380,937 | 86.40% | proteklo 1.0 s
Pregledano 13,581,359/15,380,937 | 88.30% | proteklo 1.0 s
Pregledano 13,817,154/15,380,937 | 89.83% | proteklo 1.0 s
Pregledano 14,158,136/15,380,937 | 92.05% | proteklo 1.0 s
Pregledano 14,391,939/15,380,937 | 93.57% | proteklo 1.1 s
Pregledano 14,576,913/15,380,937 | 94.77% | proteklo 1.1 s
Pregledano 14,723,124/15,380,937 | 95.72% | proteklo 1.1 s
Pregledano 14,900,027/15,380,937 | 96.87% | proteklo 1.1 s
Pregledano 15,034,820/15,380,937 | 97.75% | proteklo 1.1 s
Pregledano 15,170,346/15,380,937 | 98.63% | proteklo 1.1 s
Pregledano 15,257,650/15,380,937 | 99.20% | proteklo 1.2 s
Pregledano 15,313,805/15,380,937 | 99.56% | proteklo 1.2 s
Pregledano 15,356,428/15,380,937 | 99.84% | proteklo 1.2 s
Pregledano 15,376,987/15,380,937 | 99.97% | proteklo 1.2 s
Pregledano 15,380,937/15,380,937 | 100.00% | proteklo 1.2 s

REZULTAT: loto7_4710_k86_loto_2977.csv
NEXT: 10 x 32 y 34 z 38
Izabrana geometrijska skala: 1.50
Validacioni NLL: 17.016540088
Odvojeni test NLL: 17.134590201
Modelna verovatnoca NEXT: 6.13080257403e-07
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska potpune normalizacije: 1.554e-15
Greska marginalnih verovatnoca: 7.772e-16
Greska Qiskit izvora: 0.000e+00
Greska Qiskit interferencije: 5.551e-17
Verovatnoca izlaza 0 u interferencijskom kolu: 0.144863724238
Ova poslednja verovatnoca nije normalizovana verovatnoca Loto kombinacije.
Dijagnosticka osetljivost: 2.346899740e-02
Zavrsni projektovani gradijent: 8.869e-05
Zbir verovatnoca ukljucivanja: 7.000000000000
Verovatnoce ukljucivanja brojeva u zajednickoj NEXT raspodeli:
 1: 0.174192228
 2: 0.174661812
 3: 0.175000523
 4: 0.175264360
 5: 0.175485953
 6: 0.175682096
 7: 0.175868298
 8: 0.176051926
 9: 0.176233820
10: 0.176417863
11: 0.176615287
12: 0.176820399
13: 0.177012986
14: 0.177217858
15: 0.177431941
16: 0.177768334
17: 0.178191937
18: 0.178579061
19: 0.178809373
20: 0.178826851
21: 0.178637346
22: 0.178783177
23: 0.179553895
24: 0.180434745
25: 0.180943410
26: 0.181993994
27: 0.184497932
28: 0.186430648
29: 0.185484326
30: 0.187309149
31: 0.187963316
32: 0.177554486
33: 0.176960426
34: 0.176373399
35: 0.184883532
36: 0.191047696
37: 0.184525315
38: 0.177102207
39: 0.187388093
DIJAGNOSTIKA: Stacionarnost zavrsne optimizacije nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost optimizacije.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4710_k86_loto_plus_1733.csv
Istorijskih prelaza: 1732
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86_loto_plus_1733.csv | skala 0.75 | validacija 1
Skala: 0.75 | uzorci: 1039 | parametri: 54
Evaluacija 1/120 | cilj 17.484923225 | gradijent 1.686e+00 | proteklo 0.2 s
Evaluacija 5/120 | cilj 17.059679878 | gradijent 1.218e-01 | proteklo 0.7 s
Evaluacija 10/120 | cilj 16.866688326 | gradijent 1.075e-01 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.824105478 | gradijent 5.088e-02 | proteklo 1.7 s
Evaluacija 20/120 | cilj 16.788201848 | gradijent 1.961e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.777695670 | gradijent 1.689e-02 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.773532284 | gradijent 7.058e-03 | proteklo 3.3 s
Evaluacija 35/120 | cilj 16.769343864 | gradijent 1.061e-02 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.767555047 | gradijent 1.058e-02 | proteklo 4.4 s
Evaluacija 45/120 | cilj 16.763200716 | gradijent 1.043e-02 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.755680146 | gradijent 1.941e-02 | proteklo 5.5 s
Evaluacija 55/120 | cilj 16.752627383 | gradijent 1.415e-02 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.746437150 | gradijent 1.064e-02 | proteklo 6.6 s
Evaluacija 65/120 | cilj 16.743865972 | gradijent 4.587e-03 | proteklo 7.1 s
Evaluacija 70/120 | cilj 16.743317352 | gradijent 6.489e-03 | proteklo 7.7 s
Evaluacija 75/120 | cilj 16.743032810 | gradijent 1.872e-03 | proteklo 8.2 s
Evaluacija 80/120 | cilj 16.742438904 | gradijent 6.157e-03 | proteklo 8.8 s
Evaluacija 85/120 | cilj 16.741573534 | gradijent 3.599e-03 | proteklo 9.3 s
Najbolji cilj: 16.741298839
Max|projektovani gradijent|: 3.165e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.279061446

loto7_4710_k86_loto_plus_1733.csv | skala 0.75 | validacija 2
Skala: 0.75 | uzorci: 1212 | parametri: 54
Evaluacija 1/120 | cilj 16.819179267 | gradijent 1.540e-01 | proteklo 0.2 s
Evaluacija 5/120 | cilj 16.812727704 | gradijent 3.841e-02 | proteklo 0.6 s
Evaluacija 10/120 | cilj 16.793038269 | gradijent 2.380e-02 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.784455724 | gradijent 9.735e-03 | proteklo 1.7 s
Evaluacija 20/120 | cilj 16.780676274 | gradijent 4.844e-03 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.777551447 | gradijent 9.127e-03 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.771624991 | gradijent 1.572e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.760564580 | gradijent 1.617e-02 | proteklo 4.0 s
Evaluacija 40/120 | cilj 16.754883857 | gradijent 9.621e-03 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.751777272 | gradijent 8.443e-03 | proteklo 5.1 s
Evaluacija 50/120 | cilj 16.748380944 | gradijent 1.301e-02 | proteklo 5.6 s
Evaluacija 55/120 | cilj 16.746989284 | gradijent 4.807e-03 | proteklo 6.2 s
Evaluacija 60/120 | cilj 16.745823736 | gradijent 8.386e-03 | proteklo 6.7 s
Evaluacija 65/120 | cilj 16.742252653 | gradijent 6.903e-03 | proteklo 7.3 s
Evaluacija 70/120 | cilj 16.739146195 | gradijent 1.283e-02 | proteklo 7.8 s
Evaluacija 75/120 | cilj 16.735018853 | gradijent 6.266e-03 | proteklo 8.4 s
Evaluacija 80/120 | cilj 16.732549977 | gradijent 6.693e-03 | proteklo 8.9 s
Evaluacija 85/120 | cilj 16.731749104 | gradijent 3.462e-03 | proteklo 9.4 s
Najbolji cilj: 16.731551051
Max|projektovani gradijent|: 1.364e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.021194294

Skala 0.75 | objedinjeni validacioni NLL: 17.150127870

loto7_4710_k86_loto_plus_1733.csv | skala 1.50 | validacija 1
Skala: 1.50 | uzorci: 1039 | parametri: 54
Evaluacija 1/120 | cilj 17.189033490 | gradijent 6.103e-01 | proteklo 0.2 s
Evaluacija 5/120 | cilj 16.958194554 | gradijent 7.406e-02 | proteklo 0.6 s
Evaluacija 10/120 | cilj 16.919254497 | gradijent 1.026e-01 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.894361149 | gradijent 5.463e-02 | proteklo 1.7 s
Evaluacija 20/120 | cilj 16.877582299 | gradijent 2.271e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.859721328 | gradijent 3.546e-02 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.835324311 | gradijent 3.107e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.805792876 | gradijent 7.093e-02 | proteklo 4.0 s
Evaluacija 40/120 | cilj 16.784157550 | gradijent 4.264e-02 | proteklo 4.6 s
Evaluacija 45/120 | cilj 16.773872125 | gradijent 1.708e-02 | proteklo 5.1 s
Evaluacija 50/120 | cilj 16.761529896 | gradijent 2.450e-02 | proteklo 5.7 s
Evaluacija 55/120 | cilj 16.751938117 | gradijent 1.077e-02 | proteklo 6.2 s
Evaluacija 60/120 | cilj 16.748540763 | gradijent 1.034e-02 | proteklo 6.8 s
Evaluacija 65/120 | cilj 16.744629994 | gradijent 1.391e-02 | proteklo 7.3 s
Evaluacija 70/120 | cilj 16.739346366 | gradijent 1.540e-02 | proteklo 7.8 s
Evaluacija 75/120 | cilj 16.737111906 | gradijent 4.419e-03 | proteklo 8.4 s
Evaluacija 80/120 | cilj 16.734834241 | gradijent 8.534e-03 | proteklo 8.9 s
Evaluacija 85/120 | cilj 16.728508164 | gradijent 1.828e-02 | proteklo 9.5 s
Evaluacija 90/120 | cilj 16.724013692 | gradijent 1.822e-02 | proteklo 10.0 s
Najbolji cilj: 16.722836476
Max|projektovani gradijent|: 9.346e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.279887384

loto7_4710_k86_loto_plus_1733.csv | skala 1.50 | validacija 2
Skala: 1.50 | uzorci: 1212 | parametri: 54
Evaluacija 1/120 | cilj 16.802914784 | gradijent 5.731e-02 | proteklo 0.2 s
Evaluacija 5/120 | cilj 16.782382457 | gradijent 1.858e-02 | proteklo 0.7 s
Evaluacija 10/120 | cilj 16.767915885 | gradijent 1.568e-02 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.758460148 | gradijent 2.566e-02 | proteklo 1.7 s
Evaluacija 20/120 | cilj 16.747381619 | gradijent 1.133e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.738979650 | gradijent 1.585e-02 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.734160221 | gradijent 1.009e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.732951464 | gradijent 5.685e-03 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.730981925 | gradijent 4.628e-03 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.730438814 | gradijent 1.187e-03 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.730394750 | gradijent 7.318e-04 | proteklo 5.5 s
Evaluacija 55/120 | cilj 16.730292630 | gradijent 2.742e-03 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.729925636 | gradijent 8.265e-03 | proteklo 6.6 s
Evaluacija 65/120 | cilj 16.728737145 | gradijent 1.813e-02 | proteklo 7.1 s
Evaluacija 70/120 | cilj 16.725735113 | gradijent 1.773e-02 | proteklo 7.7 s
Evaluacija 75/120 | cilj 16.716443790 | gradijent 2.186e-02 | proteklo 8.2 s
Evaluacija 80/120 | cilj 16.708447361 | gradijent 1.190e-02 | proteklo 8.8 s
Evaluacija 85/120 | cilj 16.704330600 | gradijent 1.088e-02 | proteklo 9.3 s
Evaluacija 90/120 | cilj 16.702416492 | gradijent 3.562e-03 | proteklo 9.9 s
Najbolji cilj: 16.702416492
Max|projektovani gradijent|: 3.562e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.
Validacioni NLL: 17.219065096

Skala 1.50 | objedinjeni validacioni NLL: 17.249476240

IZBOR GEOMETRIJSKE SKALE
Skala 0.75 | validacioni NLL 17.150127870
Skala 1.50 | validacioni NLL 17.249476240
Izabrana skala: 0.75
Tolerancija pri izboru manje skale: 0.000010 NLL

loto7_4710_k86_loto_plus_1733.csv | odvojeni test
Skala: 0.75 | uzorci: 1385 | parametri: 54
Evaluacija 1/120 | cilj 16.768881053 | gradijent 4.222e-02 | proteklo 0.2 s
Evaluacija 5/120 | cilj 16.753731942 | gradijent 1.866e-02 | proteklo 0.7 s
Evaluacija 10/120 | cilj 16.744639847 | gradijent 6.868e-03 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.743706188 | gradijent 2.418e-03 | proteklo 1.8 s
Evaluacija 20/120 | cilj 16.743489037 | gradijent 1.786e-03 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.743459419 | gradijent 3.997e-04 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.743422145 | gradijent 7.225e-04 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.743395584 | gradijent 1.043e-03 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.743381108 | gradijent 2.029e-04 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.743378997 | gradijent 1.353e-04 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.743378628 | gradijent 1.583e-05 | proteklo 5.6 s
Evaluacija 55/120 | cilj 16.743378609 | gradijent 6.015e-06 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.743378607 | gradijent 1.721e-06 | proteklo 6.6 s
Najbolji cilj: 16.743378607
Max|projektovani gradijent|: 1.721e-06
Optimizator: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH

Odvojeni test NLL: 17.175103461
Test ne menja izabranu skalu.

loto7_4710_k86_loto_plus_1733.csv | zavrsni trening cele istorije
Skala: 0.75 | uzorci: 1732 | parametri: 54
Evaluacija 1/120 | cilj 16.831742242 | gradijent 4.054e-02 | proteklo 0.2 s
Evaluacija 5/120 | cilj 16.818740079 | gradijent 1.022e-02 | proteklo 0.6 s
Evaluacija 10/120 | cilj 16.813709206 | gradijent 2.815e-02 | proteklo 1.2 s
Evaluacija 15/120 | cilj 16.804841204 | gradijent 6.870e-03 | proteklo 1.7 s
Evaluacija 20/120 | cilj 16.802862732 | gradijent 1.038e-02 | proteklo 2.3 s
Evaluacija 25/120 | cilj 16.799820619 | gradijent 9.281e-03 | proteklo 2.8 s
Evaluacija 30/120 | cilj 16.796763056 | gradijent 1.170e-02 | proteklo 3.4 s
Evaluacija 35/120 | cilj 16.794002268 | gradijent 7.077e-03 | proteklo 3.9 s
Evaluacija 40/120 | cilj 16.792861842 | gradijent 3.545e-03 | proteklo 4.5 s
Evaluacija 45/120 | cilj 16.791380524 | gradijent 9.172e-03 | proteklo 5.0 s
Evaluacija 50/120 | cilj 16.788154975 | gradijent 1.263e-02 | proteklo 5.5 s
Evaluacija 55/120 | cilj 16.779397052 | gradijent 1.140e-02 | proteklo 6.1 s
Evaluacija 60/120 | cilj 16.769441762 | gradijent 1.513e-02 | proteklo 6.6 s
Evaluacija 65/120 | cilj 16.762647221 | gradijent 1.592e-02 | proteklo 7.2 s
Evaluacija 70/120 | cilj 16.758895136 | gradijent 1.010e-02 | proteklo 7.7 s
Evaluacija 75/120 | cilj 16.754901262 | gradijent 1.511e-02 | proteklo 8.3 s
Evaluacija 80/120 | cilj 16.750533648 | gradijent 9.324e-03 | proteklo 8.8 s
Evaluacija 85/120 | cilj 16.747833833 | gradijent 1.308e-02 | proteklo 9.4 s
Najbolji cilj: 16.747294117
Max|projektovani gradijent|: 6.701e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena. Budzet nije razlog za odbacivanje vazeceg kandidata.

Trazim globalni maksimum zajednicke koherentne raspodele.
Pregledano 2,022,398/15,380,937 | 13.15% | proteklo 0.1 s
Pregledano 2,685,276/15,380,937 | 17.46% | proteklo 0.2 s
Pregledano 3,364,025/15,380,937 | 21.87% | proteklo 0.2 s
Pregledano 4,702,537/15,380,937 | 30.57% | proteklo 0.3 s
Pregledano 5,070,086/15,380,937 | 32.96% | proteklo 0.3 s
Pregledano 6,126,603/15,380,937 | 39.83% | proteklo 0.4 s
Pregledano 6,923,742/15,380,937 | 45.02% | proteklo 0.5 s
Pregledano 7,299,885/15,380,937 | 47.46% | proteklo 0.5 s
Pregledano 8,392,538/15,380,937 | 54.56% | proteklo 0.6 s
Pregledano 8,653,288/15,380,937 | 56.26% | proteklo 0.6 s
Pregledano 9,623,294/15,380,937 | 62.57% | proteklo 0.7 s
Pregledano 9,988,483/15,380,937 | 64.94% | proteklo 0.7 s
Pregledano 10,720,521/15,380,937 | 69.70% | proteklo 0.7 s
Pregledano 11,090,325/15,380,937 | 72.10% | proteklo 0.8 s
Pregledano 11,703,567/15,380,937 | 76.09% | proteklo 0.8 s
Pregledano 12,007,052/15,380,937 | 78.06% | proteklo 0.8 s
Pregledano 12,572,260/15,380,937 | 81.74% | proteklo 0.9 s
Pregledano 12,751,026/15,380,937 | 82.90% | proteklo 0.9 s
Pregledano 13,289,586/15,380,937 | 86.40% | proteklo 1.0 s
Pregledano 13,581,359/15,380,937 | 88.30% | proteklo 1.0 s
Pregledano 13,817,154/15,380,937 | 89.83% | proteklo 1.0 s
Pregledano 14,158,136/15,380,937 | 92.05% | proteklo 1.0 s
Pregledano 14,391,939/15,380,937 | 93.57% | proteklo 1.1 s
Pregledano 14,576,913/15,380,937 | 94.77% | proteklo 1.1 s
Pregledano 14,723,124/15,380,937 | 95.72% | proteklo 1.1 s
Pregledano 14,900,027/15,380,937 | 96.87% | proteklo 1.1 s
Pregledano 15,034,820/15,380,937 | 97.75% | proteklo 1.1 s
Pregledano 15,170,346/15,380,937 | 98.63% | proteklo 1.1 s
Pregledano 15,257,650/15,380,937 | 99.20% | proteklo 1.2 s
Pregledano 15,313,805/15,380,937 | 99.56% | proteklo 1.2 s
Pregledano 15,356,428/15,380,937 | 99.84% | proteklo 1.2 s
Pregledano 15,376,987/15,380,937 | 99.97% | proteklo 1.2 s
Pregledano 15,380,937/15,380,937 | 100.00% | proteklo 1.2 s

REZULTAT: loto7_4710_k86_loto_plus_1733.csv
NEXT: 25 x 31 y 33 z 37
Izabrana geometrijska skala: 0.75
Validacioni NLL: 17.150127870
Odvojeni test NLL: 17.175103461
Modelna verovatnoca NEXT: 5.22951493656e-07
Zbir svih verovatnoca: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greska potpune normalizacije: 9.992e-16
Greska marginalnih verovatnoca: 7.772e-16
Greska Qiskit izvora: 0.000e+00
Greska Qiskit interferencije: 1.388e-16
Verovatnoca izlaza 0 u interferencijskom kolu: 0.136003974875
Ova poslednja verovatnoca nije normalizovana verovatnoca Loto kombinacije.
Dijagnosticka osetljivost: 5.326703068e-02
Zavrsni projektovani gradijent: 6.701e-03
Zbir verovatnoca ukljucivanja: 7.000000000000
Verovatnoce ukljucivanja brojeva u zajednickoj NEXT raspodeli:
 1: 0.189521315
 2: 0.188850516
 3: 0.188195403
 4: 0.187554278
 5: 0.186920894
 6: 0.186287996
 7: 0.185650724
 8: 0.185008504
 9: 0.184364200
10: 0.183720168
11: 0.183073986
12: 0.182419349
13: 0.181753026
14: 0.181077165
15: 0.180398751
16: 0.179733148
17: 0.179095151
18: 0.178490297
19: 0.177922206
20: 0.177420752
21: 0.177026781
22: 0.176773092
23: 0.176663205
24: 0.176556791
25: 0.176741844
26: 0.177803786
27: 0.179138475
28: 0.180287102
29: 0.182114080
30: 0.181649855
31: 0.176602149
32: 0.167826123
33: 0.162765991
34: 0.179491718
35: 0.186107741
36: 0.174432983
37: 0.156617563
38: 0.174166843
39: 0.169776050
DIJAGNOSTIKA: Stacionarnost zavrsne optimizacije nije potvrdjena.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
ZAVRSNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 2 x 22 y 30 z 36 | skala: 1.50 | test NLL: 17.049586846
loto7_4710_k86_loto_2977.csv | NEXT: 10 x 32 y 34 z 38 | skala: 1.50 | test NLL: 17.134590201
loto7_4710_k86_loto_plus_1733.csv | NEXT: 25 x 31 y 33 z 37 | skala: 0.75 | test NLL: 17.175103461
"""



"""
Interferencija koja zavisi od cele kombinacije.

Originalni rad Hannah Cairo. 
Konstrukcija koristi sume izabranih vektora, geometriju na više skala i Fourierovu analizu. 
Ključni deo je skup tačaka nastalih izborom podskupova iste veličine — to daje koristan most prema izboru sedam od 39 brojeva.

1. Svaki broj dobija vektor na zakrivljenoj geometrijskoj putanji.
2. Kombinacija dobija zajedničku koordinatu sabiranjem svojih sedam vektora.
3. Više kvantnih talasnih komponenti interferira na toj koordinati.
4. Istorija kola određuje naučene amplitude i faze kroz Qiskit.
5. Kvadrat modula zajedničke amplitude određuje težinu kombinacije; normalizacijom dobijamo raspodelu.

Sabiram kompleksne doprinose celoj kombinaciji pre računanja verovatnoće. 
Time mogu nastati zajednički obrasci koji se ne svode na proizvod sedam zasebnih izlaza.

Posebno treba proveriti da različite kombinacije ne postanu nerazlučive zbog izabrane geometrije i da zajednička raspodela može tačno da se normalizuje.

V1 kompleksni talasni doprinosi se sabiraju za celu kombinaciju, pa tek onda računaju verovatnoće. 
Prvo proveriti tačnu normalizaciju preko svih izbora sedam od 39 i način da se globalni NEXT maksimum pronađe bez nasumičnog pretraživanja. 

Matematička provera na malom primeru je prošla: normalizacija preko Fourierovih doprinosa slaže se sa potpunim nabrajanjem kombinacija, uz odstupanje oko 2.22e-16. 
Provera gradijenta dala je odstupanje oko 2.88e-10. 
To su provere formula u JavaScriptu, ne izvršavanje Python koda. 
Za NEXT koristiti deterministički pregled svih 15.380.937 kombinacija, u blokovima koji ograničavaju memoriju.

V1 čuva geometriju suma podskupova i koherentno sabiranje talasnih amplituda. 
Qiskit uči kompleksne koeficijente; verovatnoća nastaje tek posle interferencije za celu kombinaciju.
"""
