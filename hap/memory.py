"""
HDC Associative Memory — Learning Is Just Counting Co-occurrences
==================================================================
"Train online, save counts as super position. See either more ones
 or more zeros and the original integers determine summation.
 Essentially learning online."

 — your notes, capturing the essence of HDC learning

The learning rule is trivial:
    memory = Σ_i bind(percept_i, action_i)
          = XOR_i (percept_i ⊗ action_i)

Each bound pair is XORed into the consensus. This is:
    - Online: update as data arrives (no minibatch needed)
    - Single-pass: one forward pass, no backprop
    - O(D) per sample: XOR is D bits, popcount is D bits
    - Zero loss function: there is no gradient to compute

Inference:
    For new percept p:
        candidate = bind(memory, p)  (unbind)
        action = argmin_k H(candidate, action_k)
               = argmax_k sim(candidate, action_k)

"The original integers determine summation" = the individual bound
pairs don't need to be stored — their consensus carries the signal.
The Hamming distance bias reveals the correct class.

RefineHD extension (Verges Boncompte 2025, Chapter 4):
    "Adaptive Learning: RefineHD"
    Retrain misclassified samples via iterative refinement.
    Instead of retraining from scratch, we adjust the class HVs
    by re-bundling misclassified samples.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from hap.exceptions import (
    HAPMemoryError,
    HAPInferenceError,
    HAPSerializationError,
    HAPDimensionError,
)

from hap.hdc_core import (
    gen_hvs,
    hv_xor,
    hv_bind,
    hv_bundle,
    hv_consensus_sum,
    hv_majority,
    hv_hamming_sim,
    hv_popcount,
    hv_batch_sim,
    HDCConfig,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AssociativeMemory — Simple Percept→Action Binding
# ═══════════════════════════════════════════════════════════════════════════════

class AssociativeMemory:
    """
    Basic associative memory: bind(percept, action) ⊕ consensus sum.

    This is the vanilla HDC classifier from Kanerva (2009) and the paper.

    Memory = XOR_i (percept_i ⊗ action_i)

    Where:
        - percept_i is the encoded observation HV
        - action_i is the encoded action/label HV
        - ⊕ denotes consensus sum (bundle) of all bound pairs
        - Each percept_i is effectively a noisy key for action_i

    Training:
        For each (percept, action) pair:
            memory = bundle(memory, bind(percept, action))

    Inference:
        For new percept p:
            unbound = bind(memory, p)
            action = argmin_{k} H(unbound, k) for each candidate action k

    This is O(D) for both train and test.
    No backprop. No loss landscape. No learning rate.

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self._memory = torch.zeros(dim, device=self.device)
        self._n_samples = 0

    @property
    def memory(self) -> torch.Tensor:
        return self._memory

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def train(self, percept: torch.Tensor, action: torch.Tensor) -> None:
        """Train on one (percept, action) pair.

        Single-pass online update:
            memory += XOR(percept, action)

        "Save counts as super position" — the accumulation IS the learning.

        Args:
            percept: (D,) percept/observation hypervector
            action: (D,) action/label hypervector
        """
        bound = hv_bind(percept, action, self.mode)
        self._memory = self._memory + bound
        self._n_samples += 1

    def train_batch(self, percepts: torch.Tensor, actions: torch.Tensor) -> None:
        """Train on a batch of (percept, action) pairs.

        Args:
            percepts: (N, D) percepts
            actions: (N, D) actions
        """
        for p, a in zip(percepts, actions):
            self.train(p, a)

    def infer(self, percept: torch.Tensor,
              action_candidates: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Infer best-matching action for a percept.

        Unbind memory with percept and compare to action candidates.

        Args:
            percept: (D,) percept HV
            action_candidates: (K, D) candidate action HVs

        Returns:
            (best_idx, similarities) where similarities is (K,) array
        """
        if self._n_samples == 0:
            raise RuntimeError("Memory is empty — train before inference.")

        # Unbind: XOR(memory, percept) ≈ stored action
        unbound = hv_bind(self._threshold_memory(), percept, self.mode)

        # Nearest neighbor among candidates
        sims = hv_batch_sim(unbound, action_candidates)
        best_idx = sims.argmax().item()

        return best_idx, sims

    def _threshold_memory(self) -> torch.Tensor:
        """Convert accumulated float memory back to binary via majority threshold.

        "The original integers determine summation" — thresholding
        reveals the original binary pattern.

        For binary mode: threshold at > 0.5
        For bipolar mode: threshold at sign (>= 0 → +1, < 0 → -1)
        """
        return hv_majority(self._memory, mode=self.mode)

    def clear(self) -> None:
        """Reset memory."""
        self._memory.zero_()
        self._n_samples = 0

    def save(self, path: str) -> None:
        """Save memory to file."""
        torch.save({
            "memory": self._memory,
            "n_samples": self._n_samples,
            "dim": self.dim,
            "mode": self.mode,
        }, path)

    def load(self, path: str) -> None:
        """Load memory from file."""
        data = torch.load(path, map_location=self.device)
        self._memory = data["memory"]
        self._n_samples = data["n_samples"]
        self.dim = data["dim"]
        self.mode = data["mode"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ActionPerceptionMemory — Explicit Perception→Action Mapping
# ═══════════════════════════════════════════════════════════════════════════════

class ActionPerceptionMemory:
    """
    HAP core memory: stores multiple action memories as data records.

    From the paper (Section "Perception to action binding with HBVs"):
        "The memory m stores the history of the data records of a size
         w over the history of the sensing stream."

    Specifically, the memory stores:
        m = XOR_bind(encoded_velocity, encoded_images)

    where:    
        m = Σ P^k(velocity_k) ⊗ image_k   (for time steps k)

    "A data record format is created: for each time slice, we construct
     m = memory + XOR(velocity_class, image_vector)"

    This creates multiple "consensus sums" — one per velocity class.
    Inference picks the velocity class whose stored entry has the
    smallest Hamming distance to the new image vector.

    Args:
        n_classes: Number of action/velocity classes
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
    """

    def __init__(
        self,
        n_classes: int = 500,
        dim: int = 8_000,
        mode: str = "binary",
        device: Optional[str] = None,
    ):
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"

        # One memory per action class
        self._class_memories = torch.zeros(n_classes, dim, device=self.device)
        self._class_counts = torch.zeros(n_classes, dtype=torch.long, device=self.device)

    def train(self, percept: torch.Tensor, class_idx: int) -> None:
        """Store percept in the class-specific memory.

        memory[c] += percept   (percept is the encoded image)

        From the paper:
            "m = memory + XOR(velocity_class, image_vector)"
        
        The action (velocity class) acts as a key that binds the percept.
        When we query with a new percept, the memory unbinds to reveal
        which velocity classes were associated with similar percepts.

        Args:
            percept: (D,) encoded observation HV
            class_idx: Action/velocity class index (0..n_classes-1)
        """
        self._class_memories[class_idx] += percept
        self._class_counts[class_idx] += 1

    def infer(self, percept: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Find class with smallest Hamming distance to percept.

        From the paper:
            "Given an unseen image d, we compute the probability of it
             being associated with each of the basis velocity vectors v_i.
             We then choose the v_i yielding the highest probability,
             equivalent to the smallest H between m_j and d."

        Args:
            percept: (D,) encoded percept HV

        Returns:
            (best_class, similarities) where similarities is (n_classes,)
        """
        # Hamming distance to each class memory
        sims = hv_batch_sim(percept, self._class_memories)
        best_idx = sims.argmax().item()
        return best_idx, sims

    def get_velocity_class(self, query: torch.Tensor,
                           velocity_keys: torch.Tensor) -> Tuple[int, float]:
        """Alternative inference: unbind memory with velocity key.

        From the paper, Eq 4:
            "p(v_i) = P(bind(m, v_i), d)"
        
        Unbind memory with velocity key, then compare to image:
            p(v_i) = 1 - H_n(bind(m, v_i), d)

        Args:
            query: (D,) image HV
            velocity_keys: (K, D) velocity class HVs

        Returns:
            (best_idx, probability)
        """
        best_prob = -1.0
        best_idx = 0

        for i, v_key in enumerate(velocity_keys):
            # Unbind: bind(m, v_i) ≈ image that was paired with velocity v_i
            unbound = hv_bind(self._threshold_memory(), v_key, self.mode)
            prob = hv_hamming_sim(unbound, query)
            if prob > best_prob:
                best_prob = prob
                best_idx = i

        return best_idx, best_prob

    def _threshold_memory(self) -> torch.Tensor:
        """Threshold accumulated memory to binary per class.

        The consensus memory per class is the bundle:
            memory[c] = majority(mean(percepts_for_class_c))
        
        Returns per-class thresholded memories stacked as (K, D).
        """
        counts = self._class_counts.unsqueeze(-1).clamp(min=1)
        return hv_majority(self._class_memories / counts.float(), mode=self.mode)

    def clear(self) -> None:
        self._class_memories.zero_()
        self._class_counts.zero_()

    def save(self, path: str) -> None:
        torch.save({
            "class_memories": self._class_memories,
            "class_counts": self._class_counts,
            "n_classes": self.n_classes,
            "dim": self.dim,
            "mode": self.mode,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self._class_memories = data["class_memories"]
        self._class_counts = data["class_counts"]
        self.n_classes = data["n_classes"]
        self.dim = data["dim"]
        self.mode = data["mode"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DataRecordMemory — Sliding Window Data Records
# ═══════════════════════════════════════════════════════════════════════════════

class DataRecordMemory:
    """
    Sliding window memory that stores recent data records as bound pairs.

    From the paper (Section "Properties of HBVs", item 5):
        "A memory m has the capacity to store about D records.
         Under the conditions of our experiments with D = 10,000,
         this means we can store around 700 records before
         statistically significant matches break down."

    Each record is: bind(percept, action) with a temporal permutation.
    The memory is the consensus sum of the sliding window.

    "Time based hierarchical representation of experiences. Train online,
     save counts as super position."

    Args:
        window_size: Max number of records to store (paper: ~700 for D=10,000)
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
    """

    def __init__(
        self,
        window_size: int = 700,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self._memory = torch.zeros(dim, device=self.device)
        self._records: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._n_records = 0

    @property
    def memory(self) -> torch.Tensor:
        return self._memory

    @property
    def is_full(self) -> bool:
        return self._n_records >= self.window_size

    def add(self, percept: torch.Tensor, action: torch.Tensor) -> None:
        """Add a new (percept, action) record.

        If at capacity, remove oldest record first.

        Args:
            percept: (D,) percept HV
            action: (D,) action HV
        """
        if self.is_full:
            # Remove oldest: subtract its contribution
            oldest_p, oldest_a = self._records.pop(0)
            bound = hv_bind(oldest_p, oldest_a, self.mode)
            self._memory = self._memory - bound
            self._n_records -= 1

        # Add new record
        self._records.append((percept.clone(), action.clone()))
        bound = hv_bind(percept, action, self.mode)
        self._memory = self._memory + bound
        self._n_records += 1

    def infer(self, percept: torch.Tensor,
              action_candidates: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Infer action for percept using thresholded memory.

        Args:
            percept: (D,) percept HV
            action_candidates: (K, D) candidate action HVs

        Returns:
            (best_idx, similarities)
        """
        unbound = hv_bind(self._threshold_memory(), percept, self.mode)
        sims = hv_batch_sim(unbound, action_candidates)
        best_idx = sims.argmax().item()
        return best_idx, sims

    def _threshold_memory(self) -> torch.Tensor:
        """Binary threshold the accumulated sliding-window memory."""
        return hv_majority(self._memory, mode=self.mode)

    def clear(self) -> None:
        self._memory.zero_()
        self._records.clear()
        self._n_records = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCClassifier — Multi-Class Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class HDCClassifier:
    """
    Full multi-class HDC classifier with class-specific label HVs.

    For each class k:
        class_hv[k] = bound random HV (the class prototype)
        memory[k] = Σ XOR(percept_i, class_hv[class_i]) for all samples i

    Inference:
        For percept p:
            unbound = XOR(memory, p)
            class = argmax_k Hamming(unbound, class_hv[k])

    Args:
        n_classes: Number of classes
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
    """

    def __init__(
        self,
        n_classes: int,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"

        # One random HV per class (acts as class signature)
        self.class_hvs = gen_hvs(n_classes, dim, mode, self.device, seed or 42)

        # One associative memory per class
        self.memory = AssociativeMemory(dim, mode, self.device)

        # Label cache for training convenience
        self._labels: List[int] = []

    def fit(self, percepts: torch.Tensor, labels: List[int]) -> None:
        """Train on a dataset.

        All percepts are bound to their class HV and bundled.

        Args:
            percepts: (N, D) percept HVs
            labels: (N,) class labels as integers (0..n_classes-1)
        """
        self.memory.clear()
        self._labels.clear()

        for p, lbl in zip(percepts, labels):
            class_hv = self.class_hvs[lbl]
            # Store label for potential refinement
            self._labels.append(lbl)
            self.memory.train(p, class_hv)

    def predict(self, percept: torch.Tensor) -> int:
        """Predict class for a single percept.

        Args:
            percept: (D,) percept HV

        Returns:
            Predicted class index (0..n_classes-1)
        """
        best_idx, _ = self.memory.infer(percept, self.class_hvs)
        return best_idx

    def predict_batch(self, percepts: torch.Tensor) -> List[int]:
        """Predict class for a batch of percepts.

        Args:
            percepts: (N, D) percept HVs

        Returns:
            (N,) list of predicted class indices
        """
        predictions = []
        for p in percepts:
            predictions.append(self.predict(p))
        return predictions

    def accuracy(self, percepts: torch.Tensor,
                 true_labels: List[int]) -> float:
        """Compute classification accuracy.

        Args:
            percepts: (N, D) percept HVs
            true_labels: (N,) true class labels

        Returns:
            Accuracy in [0, 1]
        """
        preds = self.predict_batch(percepts)
        correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
        return correct / len(true_labels)

    def save(self, path: str) -> None:
        torch.save({
            "class_hvs": self.class_hvs,
            "memory": self.memory._memory,
            "n_samples": self.memory._n_samples,
            "n_classes": self.n_classes,
            "dim": self.dim,
            "mode": self.mode,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.class_hvs = data["class_hvs"]
        self.memory._memory = data["memory"]
        self.memory._n_samples = data["n_samples"]
        self.n_classes = data["n_classes"]
        self.dim = data["dim"]
        self.mode = data["mode"]

    def clear(self) -> None:
        self.memory.clear()
        self._labels.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RefineHDLearner — Adaptive Refinement
# ═══════════════════════════════════════════════════════════════════════════════

class RefineHDLearner:
    """
    RefineHD: Iterative refinement for misclassified samples.

    From Verges Boncompte (2025), Chapter 4:
        "RefineHD: Adaptive Learning for HD Computing."

    The idea: misclassified samples are re-bundled into the correct
    class memory with adjusted weights. This corrects the prototype
    without full retraining.

    Algorithm:
        1. Train initial classifier (single pass)
        2. Evaluate on training set, identify misclassified samples
        3. For each misclassified sample:
            - Re-bind to correct class with higher weight
            - Re-bind to wrong class with negative weight (if possible)
        4. Repeat for n_refinement rounds

    This refines the decision boundary without backpropagation.

    Args:
        classifier: HDCClassifier instance
        n_refinement_rounds: Number of refinement iterations (default: 3)
        refinement_weight: Weight for misclassified samples (default: 2.0)
    """

    def __init__(
        self,
        classifier: HDCClassifier,
        n_refinement_rounds: int = 3,
        refinement_weight: float = 2.0,
    ):
        self.classifier = classifier
        self.n_rounds = n_refinement_rounds
        self.refinement_weight = refinement_weight
        self._history: List[Dict] = []

    def fit(self, percepts: torch.Tensor, labels: List[int]) -> Dict:
        """Train with refinement.

        Args:
            percepts: (N, D) percept HVs
            labels: (N,) true labels

        Returns:
            Dict with accuracy history per round
        """
        dim = percepts.shape[-1]
        device = percepts.device

        # Step 1: Initial training
        self.classifier.fit(percepts, labels)
        init_acc = self.classifier.accuracy(percepts, labels)
        self._history = [{"round": 0, "accuracy": init_acc}]

        # Step 2: Refinement rounds
        for rnd in range(1, self.n_rounds + 1):
            misclassified = 0

            for p, true_lbl in zip(percepts, labels):
                pred_lbl = self.classifier.predict(p)

                if pred_lbl != true_lbl:
                    # Re-bind to correct class with higher weight
                    correct_hv = self.classifier.class_hvs[true_lbl]
                    wrong_hv = self.classifier.class_hvs[pred_lbl]

                    bound_correct = hv_bind(p, correct_hv, self.classifier.mode)
                    bound_wrong = hv_bind(p, wrong_hv, self.classifier.mode)

                    # Add weighted correction to memory
                    # Increase correct class, decrease wrong class
                    self.classifier.memory._memory += (
                        self.refinement_weight * bound_correct
                    )
                    self.classifier.memory._memory -= (
                        (self.refinement_weight - 1.0) * bound_wrong
                    )
                    misclassified += 1

            acc = self.classifier.accuracy(percepts, labels)
            self._history.append({
                "round": rnd,
                "accuracy": acc,
                "misclassified": misclassified,
            })

        return {
            "initial_accuracy": init_acc,
            "final_accuracy": acc,
            "history": self._history,
        }
