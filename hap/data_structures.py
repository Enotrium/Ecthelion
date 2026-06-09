"""
HDC Data Structure Representations — Graphs, Trees, FSA, N-Grams
===============================================================
Extends the data record / set / sequence primitives from the paper
into more complex relational abstractions.

From Kleyko et al., "A Comprehensive Study of Complexity and Performance
of Automatic Detection of Atrial Fibrillation: Classification with
Hyperdimensional Computing" and the HDC/VSA Data Structures Cookbook.

All structures reduce to the same operations:
    bind(percept, role) + bundle

Data structures supported:
    - Directed/Undirected Graphs: edge binding + role permutations
    - Trees: hierarchical path encoding via role-filler binding
    - Finite State Automata: transition mapping
    - N-Gram Statistics: sequential binding + accumulation
    - Frequency Distributions: weighted bundling
    - Stacks: permutation-based shift LIFO

Each structure supports:
    - Encoding: build the HD representation
    - Querying: extract subcomponents via unbind + nearest-neighbor
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from hap.hdc_core import (
    gen_hvs,
    hv_xor,
    hv_bind,
    hv_bundle,
    hv_permute,
    hv_batch_sim,
    hv_hamming_sim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GraphEncoder — Directed & Undirected Graphs
# ═══════════════════════════════════════════════════════════════════════════════

class GraphEncoder:
    """Encode graphs as HVs via edge-binding + role permutations.

    From the cookbook (DS.py, graph function):
        Undirected:  H(e) = bind(H(v1), H(v2))
        Directed:    H(e) = bind(H(v1), permute(H(v2)))
        Graph HV = bundle([H(e1), H(e2), ..., H(en)])

    Queries:
        - Find outgoing connections from v: unbind(graph, H(v))
        - Find incoming connections to v: unbind(graph, permute(H(v)))

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        # Vocabulary: maps vertex/key names to HVs
        self._vertices: Dict[str, torch.Tensor] = {}
        self._seed_counter = seed or 42

    def add_vertex(self, name: str) -> torch.Tensor:
        """Register a vertex, generating a random HV for it.

        Args:
            name: Vertex identifier

        Returns:
            (dim,) vertex HV
        """
        if name not in self._vertices:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._vertices),
            ).squeeze(0)
            self._vertices[name] = hv
        return self._vertices[name]

    def encode(
        self,
        edges: List[Tuple[str, str]],
        directed: bool = False,
    ) -> torch.Tensor:
        """Encode a graph into a single HV.

        Args:
            edges: List of (v1, v2) tuples — v1 connects to v2
            directed: If True, edge direction matters (outgoing vs incoming)

        Returns:
            (dim,) graph hypervector
        """
        edge_hvs = []
        for v1_name, v2_name in edges:
            hv1 = self.add_vertex(v1_name)
            hv2 = self.add_vertex(v2_name)

            if directed:
                # bind(v1, permute(v2)) — direction encoded in permutation
                edge_hv = hv_bind(hv1, hv_permute(hv2), self.mode)
            else:
                # bind(v1, v2) — symmetric
                edge_hv = hv_bind(hv1, hv2, self.mode)

            edge_hvs.append(edge_hv)

        if not edge_hvs:
            return torch.zeros(self.dim, device=self.device)

        return hv_bundle(torch.stack(edge_hvs))

    def outgoing(self, graph_hv: torch.Tensor,
                 vertex_name: str,
                 candidates: List[str]) -> List[Tuple[str, float]]:
        """Find outgoing connections from a vertex.

        For undirected: unbind(graph, v) ≈ sum of connected vertices
        For directed: unbind(graph, v) ≈ sum of target vertices

        Args:
            graph_hv: (dim,) encoded graph
            vertex_name: Source vertex to query
            candidates: List of candidate destination vertex names

        Returns:
            List of (vertex_name, similarity) sorted decreasing
        """
        hv = self._vertices.get(vertex_name)
        if hv is None:
            raise KeyError(f"Unknown vertex: {vertex_name}")

        unbound = hv_bind(graph_hv, hv, self.mode)

        sims = []
        for c_name in candidates:
            if c_name != vertex_name and c_name in self._vertices:
                c_hv = self._vertices[c_name]
                sim = hv_hamming_sim(unbound, c_hv).item()
                sims.append((c_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def incoming(self, graph_hv: torch.Tensor,
                 vertex_name: str,
                 candidates: List[str]) -> List[Tuple[str, float]]:
        """Find incoming connections to a vertex (directed only).

        unbind(graph, permute(v)) ≈ sum of source vertices

        Args:
            graph_hv: (dim,) encoded graph
            vertex_name: Target vertex to query
            candidates: List of candidate source vertex names

        Returns:
            List of (vertex_name, similarity) sorted decreasing
        """
        hv = self._vertices.get(vertex_name)
        if hv is None:
            raise KeyError(f"Unknown vertex: {vertex_name}")

        # Undo the forward permutation
        unbound = hv_bind(graph_hv, hv_permute(hv), self.mode)

        # For incoming, reverse the permutation
        unbound = hv_permute(unbound, -1)

        sims = []
        for c_name in candidates:
            if c_name != vertex_name and c_name in self._vertices:
                c_hv = self._vertices[c_name]
                sim = hv_hamming_sim(unbound, c_hv).item()
                sims.append((c_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    @property
    def vertices(self) -> List[str]:
        return list(self._vertices.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TreeEncoder — Hierarchical Path → Symbol
# ═══════════════════════════════════════════════════════════════════════════════

class TreeEncoder:
    """Encode binary trees via role-filler binding on paths.

    From the cookbook (DS.py, tree function):
        For each leaf (symbol, path):
            path_HV = encode_sequence([role_left, role_right, ...])
            record = bind(path_HV, symbol_HV)
        Tree HV = bundle(all leaf records)

    Queries:
        - Symbol at path: unbind(tree, path_HV) ≈ symbol_HV
        - Path of symbol: unbind(tree, symbol_HV) ≈ path_HV

    Roles are 'L' (left) and 'R' (right) by default.

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        # Role HVs for left and right
        self._role_left = gen_hvs(1, dim, mode, self.device, self.seed).squeeze(0)
        self._role_right = gen_hvs(1, dim, mode, self.device, self.seed + 1).squeeze(0)

        # Symbol HVs (created on demand)
        self._symbols: Dict[str, torch.Tensor] = {}
        self._seed_counter = self.seed + 10

    def add_symbol(self, name: str) -> torch.Tensor:
        """Register a symbol HV."""
        if name not in self._symbols:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._symbols),
            ).squeeze(0)
            self._symbols[name] = hv
        return self._symbols[name]

    def _encode_path(self, roles: Sequence[str]) -> torch.Tensor:
        """Encode a path sequence [L, R, L, ...] into an HV.

        Uses binding-based sequence encoding:
            path_HV = bind(P^(n-1)(r1), P^(n-2)(r2), ..., rn)

        This creates a unique representation for each distinct path.
        """
        role_map = {"L": self._role_left, "R": self._role_right}
        hvs = []
        for role in roles:
            if role not in role_map:
                raise ValueError(f"Unknown role: {role}. Use 'L' or 'R'.")
            hvs.append(role_map[role])

        if not hvs:
            return torch.zeros(self.dim, device=self.device)

        # Binding-based sequence encoding
        n = len(hvs)
        path_hv = hv_permute(hvs[0], n - 1)
        for i in range(1, n):
            shifted = hv_permute(hvs[i], n - 1 - i)
            path_hv = hv_bind(path_hv, shifted, self.mode)

        return path_hv

    def encode(self, tree_entries: List[Tuple[str, List[str]]]) -> torch.Tensor:
        """Encode a tree from (symbol, path) entries.

        Args:
            tree_entries: List of (symbol_name, [role1, role2, ...])
                         e.g., [("a", ["L","L","L"]), ("b", ["L","R"])]

        Returns:
            (dim,) tree hypervector
        """
        records = []
        for symbol_name, path in tree_entries:
            sym_hv = self.add_symbol(symbol_name)
            path_hv = self._encode_path(path)
            record = hv_bind(path_hv, sym_hv, self.mode)
            records.append(record)

        if not records:
            return torch.zeros(self.dim, device=self.device)

        return hv_bundle(torch.stack(records))

    def symbol_at_path(self, tree_hv: torch.Tensor,
                       path: List[str],
                       candidates: List[str]) -> List[Tuple[str, float]]:
        """Find the symbol at a given path.

        unbind(tree, path_HV) ≈ symbol_HV

        Args:
            tree_hv: (dim,) encoded tree
            path: Path as list of roles e.g. ['L', 'R', 'L']
            candidates: Candidate symbol names

        Returns:
            List of (symbol_name, similarity) sorted decreasing
        """
        path_hv = self._encode_path(path)
        unbound = hv_bind(tree_hv, path_hv, self.mode)

        sims = []
        for c_name in candidates:
            if c_name in self._symbols:
                c_hv = self._symbols[c_name]
                sim = hv_hamming_sim(unbound, c_hv).item()
                sims.append((c_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def path_of_symbol(self, tree_hv: torch.Tensor,
                       symbol_name: str) -> torch.Tensor:
        """Recover the approximate path HV for a symbol.

        unbind(tree, symbol_HV) ≈ path_HV

        Args:
            tree_hv: (dim,) encoded tree
            symbol_name: Symbol to find path for

        Returns:
            (dim,) approximate path HV
        """
        sym_hv = self._symbols.get(symbol_name)
        if sym_hv is None:
            raise KeyError(f"Unknown symbol: {symbol_name}")

        return hv_bind(tree_hv, sym_hv, self.mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FSAEncoder — Finite State Automata
# ═══════════════════════════════════════════════════════════════════════════════

class FSAEncoder:
    """Encode finite state automata transition mappings.

    From the cookbook (DS.py, fsa function):
        For each transition (state_from, state_to, input):
            state_seq = bind(P(state_from), state_to)  [bidirectional pair]
            transition_HV = bind(state_seq, input_HV)
        FSA HV = bundle(all transitions)

    Query:
        Given (state, input):
            unbind(unbind(fsa, input_HV), state_HV)
            de-permute → next_state_HV

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        self._states: Dict[str, torch.Tensor] = {}
        self._inputs: Dict[str, torch.Tensor] = {}
        self._seed_counter = self.seed + 100

    def add_state(self, name: str) -> torch.Tensor:
        """Register a state HV."""
        if name not in self._states:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._states),
            ).squeeze(0)
            self._states[name] = hv
        return self._states[name]

    def add_input(self, name: str) -> torch.Tensor:
        """Register an input symbol HV."""
        if name not in self._inputs:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + 1000 + len(self._inputs),
            ).squeeze(0)
            self._inputs[name] = hv
        return self._inputs[name]

    def encode(
        self,
        transitions: List[Tuple[str, str, str]],
    ) -> torch.Tensor:
        """Encode FSA transitions.

        Args:
            transitions: List of (state_from, state_to, input_symbol)
                        e.g., [("Lock","Lock","Push"), ("Lock","Unlock","Token")]

        Returns:
            (dim,) FSA hypervector
        """
        trans_hvs = []
        for state_from, state_to, input_sym in transitions:
            hv_from = self.add_state(state_from)
            hv_to = self.add_state(state_to)
            hv_input = self.add_input(input_sym)

            # Encode state pair as a binding-based sequence
            state_seq = hv_bind(hv_permute(hv_from), hv_to, self.mode)

            # Bind with input
            trans_hv = hv_bind(state_seq, hv_input, self.mode)
            trans_hvs.append(trans_hv)

        if not trans_hvs:
            return torch.zeros(self.dim, device=self.device)

        return hv_bundle(torch.stack(trans_hvs))

    def next_state(self, fsa_hv: torch.Tensor,
                   current_state: str,
                   input_sym: str,
                   candidates: List[str]) -> List[Tuple[str, float]]:
        """Query: given state and input, find the next state.

        unbind(unbind(fsa, input_HV), state_HV) ≈ permute(next_state_HV)
        de-permute → compare with candidates

        Args:
            fsa_hv: (dim,) encoded FSA
            current_state: Current state name
            input_sym: Input symbol name
            candidates: Candidate next state names

        Returns:
            List of (state_name, similarity) sorted decreasing
        """
        hv_state = self._states.get(current_state)
        hv_input = self._inputs.get(input_sym)
        if hv_state is None:
            raise KeyError(f"Unknown state: {current_state}")
        if hv_input is None:
            raise KeyError(f"Unknown input: {input_sym}")

        # Step 1: unbind input to recover approximate state_seq bundles
        unbound_input = hv_bind(fsa_hv, hv_input, self.mode)

        # Step 2: unbind the PERMUTED current state (matches encoding: bind(P(from), to))
        unbound_state = hv_bind(unbound_input, hv_permute(hv_state), self.mode)

        # Step 3: result is approximate "to" state directly
        approx_next = unbound_state

        # Step 4: match against candidates
        sims = []
        for c_name in candidates:
            if c_name != current_state and c_name in self._states:
                c_hv = self._states[c_name]
                sim = hv_hamming_sim(approx_next, c_hv).item()
                sims.append((c_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NGramEncoder — Sequential N-Gram Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class NGramEncoder:
    """Encode n-gram statistics from a sequence of symbols.

    From the cookbook (DS.py, ngram function):
        For each position i in data:
            ngram_rep = encode_sequence(data[i:i+n])  [binding-based]
            accumulator += ngram_rep
        Result = bundle(accumulator)

    Similar sequences have similar n-gram distribution HVs.
    The accumulator captures the frequency of each n-gram.

    Args:
        dim: HV dimensionality
        n: n-gram size (e.g., 3 for trigrams)
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        n: int = 3,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n = n
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        self._symbols: Dict[str, torch.Tensor] = {}
        self._seed_counter = self.seed + 200

    def add_symbol(self, name: str) -> torch.Tensor:
        """Register a symbol HV."""
        if name not in self._symbols:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._symbols),
            ).squeeze(0)
            self._symbols[name] = hv
        return self._symbols[name]

    def encode(self, sequence: List[str]) -> torch.Tensor:
        """Encode n-gram statistics for a sequence.

        Args:
            sequence: List of symbol names

        Returns:
            (dim,) n-gram statistics HV
        """
        if len(sequence) < self.n:
            return torch.zeros(self.dim, device=self.device)

        accumulator = torch.zeros(self.dim, device=self.device)

        for i in range(len(sequence) - self.n + 1):
            ngram = sequence[i:i + self.n]
            ngram_hv = self._encode_ngram(ngram)
            accumulator = accumulator + ngram_hv

        # Bundle to threshold
        return hv_bundle(accumulator.unsqueeze(0))

    def _encode_ngram(self, symbols: List[str]) -> torch.Tensor:
        """Encode a single n-gram via binding-based sequence.

        ngram_HV = bind(P^(n-1)(s1), P^(n-2)(s2), ..., sn)
        """
        hvs = [self.add_symbol(s) for s in symbols]
        n = len(hvs)

        result = hv_permute(hvs[0], n - 1)
        for i in range(1, n):
            shifted = hv_permute(hvs[i], n - 1 - i)
            result = hv_bind(result, shifted, self.mode)

        return result

    def similarity_between(self, seq1: List[str],
                           seq2: List[str]) -> float:
        """Compute n-gram distribution similarity between two sequences.

        Args:
            seq1, seq2: Two symbol sequences

        Returns:
            Hamming similarity in [0, 1]
        """
        hv1 = self.encode(seq1)
        hv2 = self.encode(seq2)
        return hv_hamming_sim(hv1, hv2).item()

    def contains_ngram(self, stats_hv: torch.Tensor,
                       ngram: List[str]) -> float:
        """Check if an n-gram is present in the statistics.

        Args:
            stats_hv: (dim,) accumulated n-gram statistics HV
            ngram: List of n symbols

        Returns:
            Similarity in [0, 1] — higher means present
        """
        ngram_hv = self._encode_ngram(ngram)
        return hv_hamming_sim(stats_hv, ngram_hv).item()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FrequencyEncoder — Weighted Frequency Distributions
# ═══════════════════════════════════════════════════════════════════════════════

class FrequencyEncoder:
    """Encode frequency distributions over symbols.

    From the cookbook (DS.py, frequency function):
        For each (symbol, frequency) pair:
            scaled_hv = frequency * symbol_HV
        Distribution HV = bundle(all scaled HVs)

    The resulting HV encodes both which symbols are present
    and their relative frequencies.

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        self._symbols: Dict[str, torch.Tensor] = {}
        self._seed_counter = self.seed + 300

    def add_symbol(self, name: str) -> torch.Tensor:
        """Register a symbol HV."""
        if name not in self._symbols:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._symbols),
            ).squeeze(0)
            self._symbols[name] = hv
        return self._symbols[name]

    def encode(self, symbol_freqs: Dict[str, float]) -> torch.Tensor:
        """Encode a frequency distribution.

        Args:
            symbol_freqs: {symbol_name: frequency}

        Returns:
            (dim,) frequency distribution HV
        """
        scaled_hvs = []
        for name, freq in symbol_freqs.items():
            hv = self.add_symbol(name)
            scaled = freq * hv
            scaled_hvs.append(scaled)

        if not scaled_hvs:
            return torch.zeros(self.dim, device=self.device)

        return hv_bundle(torch.stack(scaled_hvs))

    def rank_symbols(self, dist_hv: torch.Tensor,
                     candidates: List[str]) -> List[Tuple[str, float]]:
        """Rank symbols by their similarity to the distribution.

        Args:
            dist_hv: (dim,) frequency distribution HV
            candidates: Candidate symbol names

        Returns:
            List of (symbol_name, similarity) sorted by frequency
        """
        sims = []
        for c_name in candidates:
            if c_name in self._symbols:
                c_hv = self._symbols[c_name]
                sim = hv_hamming_sim(dist_hv, c_hv).item()
                sims.append((c_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims


# ═══════════════════════════════════════════════════════════════════════════════
# 6. StackEncoder — LIFO Stack via Permutation
# ═══════════════════════════════════════════════════════════════════════════════

class StackEncoder:
    """Encode a LIFO stack using permutation-based position encoding.

    From the cookbook (DS.py, stack example):
        Stack = bundle([s1, P(s2), P^2(s3), ...])
        where s1 is the top.

    Operations:
        Push:  stack' = bundle([new, P(stack)])
        Pop:   identify top via nearest-neighbor, then P^(-1)(stack - top)

    Args:
        dim: HV dimensionality
        mode: 'binary' or 'bipolar'
        device: torch device
        seed: Random seed
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.mode = mode
        self.device = device or "cpu"
        self.seed = seed or 42

        self._symbols: Dict[str, torch.Tensor] = {}
        self._seed_counter = self.seed + 400

    def add_symbol(self, name: str) -> torch.Tensor:
        """Register a symbol HV."""
        if name not in self._symbols:
            hv = gen_hvs(
                1, self.dim, self.mode, self.device,
                self._seed_counter + len(self._symbols),
            ).squeeze(0)
            self._symbols[name] = hv
        return self._symbols[name]

    def encode(self, items: List[str]) -> torch.Tensor:
        """Encode a stack of items.

        items[0] is top of stack.
        items[-1] is bottom of stack.

        Args:
            items: List of symbol names (top first)

        Returns:
            (dim,) stack HV
        """
        if not items:
            return torch.zeros(self.dim, device=self.device)

        hvs = []
        for depth, item in enumerate(items):
            hv = self.add_symbol(item)
            # Shift by depth: top=0, items deeper get more permutation
            shifted = hv_permute(hv, depth)
            hvs.append(shifted)

        return hv_bundle(torch.stack(hvs))

    def push(self, stack_hv: torch.Tensor,
             item: str) -> torch.Tensor:
        """Push an item onto the stack.

        stack' = bundle([item_hv, P(stack)])

        Args:
            stack_hv: (dim,) current stack HV
            item: Item to push

        Returns:
            (dim,) new stack HV
        """
        item_hv = self.add_symbol(item)
        shifted_stack = hv_permute(stack_hv)
        return hv_bundle(torch.stack([item_hv, shifted_stack]))

    def peek(self, stack_hv: torch.Tensor,
             candidates: List[str]) -> Tuple[str, float]:
        """Identify the top item on the stack.

        The top item shares nearest-neighbor similarity with the stack HV.

        Args:
            stack_hv: (dim,) current stack HV
            candidates: Candidate symbol names

        Returns:
            (symbol_name, confidence) of the top item
        """
        best_sim = -1.0
        best_name = ""

        for c_name in candidates:
            if c_name in self._symbols:
                c_hv = self._symbols[c_name]
                sim = hv_hamming_sim(stack_hv, c_hv).item()
                if sim > best_sim:
                    best_sim = sim
                    best_name = c_name

        return best_name, best_sim

    def pop(self, stack_hv: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Pop the top item from the stack.

        Step 1: Identify top via nearest-neighbor
        Step 2: Remove top: stack' = P^(-1)(stack - top_HV)

        Args:
            stack_hv: (dim,) current stack HV

        Returns:
            (popped_item_name, new_stack_hv)
        """
        # Identify top
        top_name, _ = self.peek(stack_hv, list(self._symbols.keys()))
        top_hv = self._symbols[top_name]

        # Remove top and de-permute
        removed = stack_hv - top_hv
        new_stack = hv_permute(removed, -1)

        return top_name, new_stack