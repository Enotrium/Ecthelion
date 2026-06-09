"""Tests for hap.data_structures — Graphs, Trees, FSA, N-Grams, Frequency, Stacks."""

import pytest
import torch
from hap.data_structures import (
    GraphEncoder,
    TreeEncoder,
    FSAEncoder,
    NGramEncoder,
    FrequencyEncoder,
    StackEncoder,
)


class TestGraphEncoder:
    def test_add_vertex(self):
        g = GraphEncoder(dim=1000, seed=42)
        hv = g.add_vertex("a")
        assert hv.shape == (1000,)
        assert "a" in g.vertices

    def test_encode_undirected(self):
        g = GraphEncoder(dim=2000, seed=42)
        edges = [("a", "b"), ("a", "c"), ("b", "d")]
        graph = g.encode(edges, directed=False)
        assert graph.shape == (2000,)

    def test_encode_directed(self):
        g = GraphEncoder(dim=2000, seed=42)
        edges = [("a", "b"), ("b", "c")]
        graph = g.encode(edges, directed=True)
        assert graph.shape == (2000,)

    def test_outgoing_query(self):
        g = GraphEncoder(dim=5000, seed=42)
        edges = [("a", "b"), ("a", "c"), ("b", "d")]
        graph = g.encode(edges, directed=False)
        results = g.outgoing(graph, "a", ["b", "c", "d"])
        # "b" and "c" should have higher similarity than "d"
        assert results[0][0] in ("b", "c")
        assert results[1][0] in ("b", "c")
        assert results[2][0] == "d"
        assert results[0][1] > results[2][1]

    def test_incoming_query(self):
        g = GraphEncoder(dim=5000, seed=42)
        edges = [("a", "b"), ("c", "b"), ("d", "e")]
        graph = g.encode(edges, directed=True)
        results = g.incoming(graph, "b", ["a", "c", "d"])
        assert results[0][0] in ("a", "c")
        assert results[1][0] in ("a", "c")
        assert results[0][1] > results[2][1]

    def test_empty_graph(self):
        g = GraphEncoder(dim=1000)
        graph = g.encode([])
        assert graph.sum().item() == 0

    def test_unknown_vertex_query(self):
        g = GraphEncoder(dim=1000)
        with pytest.raises(KeyError):
            g.outgoing(torch.ones(1000), "nonexistent", ["a"])


class TestTreeEncoder:
    def test_encode_tree(self):
        t = TreeEncoder(dim=4000, seed=42)
        entries = [
            ("a", ["L", "L", "L"]),
            ("b", ["L", "R"]),
            ("c", ["R", "R", "L"]),
        ]
        tree = t.encode(entries)
        assert tree.shape == (4000,)

    def test_symbol_at_path(self):
        t = TreeEncoder(dim=5000, seed=42)
        entries = [
            ("apple", ["L", "L"]),
            ("banana", ["L", "R"]),
            ("cherry", ["R", "L"]),
        ]
        tree = t.encode(entries)
        results = t.symbol_at_path(tree, ["L", "L"], ["apple", "banana", "cherry"])
        assert results[0][0] == "apple"
        assert results[0][1] > results[1][1]

    def test_path_of_symbol(self):
        t = TreeEncoder(dim=5000, seed=42)
        entries = [
            ("apple", ["L", "L"]),
            ("banana", ["L", "R"]),
        ]
        tree = t.encode(entries)
        path_hv = t.path_of_symbol(tree, "apple")
        assert path_hv.shape == (5000,)

    def test_empty_tree(self):
        t = TreeEncoder(dim=1000)
        tree = t.encode([])
        assert tree.sum().item() == 0

    def test_unknown_role(self):
        t = TreeEncoder(dim=1000)
        with pytest.raises(ValueError):
            t._encode_path(["X"])

    def test_unknown_symbol_query(self):
        t = TreeEncoder(dim=1000)
        with pytest.raises(KeyError):
            t.path_of_symbol(torch.ones(1000), "nonexistent")


class TestFSAEncoder:
    def test_encode_transitions(self):
        fsa = FSAEncoder(dim=4000, seed=42)
        transitions = [
            ("Lock", "Lock", "Push"),
            ("Lock", "Unlock", "Token"),
            ("Unlock", "Unlock", "Push"),
            ("Unlock", "Lock", "Token"),
        ]
        fsa_hv = fsa.encode(transitions)
        assert fsa_hv.shape == (4000,)

    def test_next_state_in_candidates(self):
        fsa = FSAEncoder(dim=8000, seed=42)
        # Register extra states so they appear as candidates
        fsa.add_state("Foo")
        fsa.add_state("Bar")
        transitions = [
            ("Lock", "Lock", "Push"),
            ("Lock", "Unlock", "Token"),
        ]
        fsa_hv = fsa.encode(transitions)

        # From Lock + Token → Unlock should rank top among multiple candidates
        results = fsa.next_state(
            fsa_hv, "Lock", "Token",
            ["Unlock", "Foo", "Bar"],
        )
        assert len(results) >= 2
        assert results[0][0] == "Unlock"
        assert results[0][1] > 0.5  # Confident match

    def test_unknown_state(self):
        fsa = FSAEncoder(dim=1000)
        with pytest.raises(KeyError):
            fsa.next_state(torch.ones(1000), "nonexistent", "Push", ["Lock"])

    def test_unknown_input(self):
        fsa = FSAEncoder(dim=1000)
        fsa.add_state("Lock")
        with pytest.raises(KeyError):
            fsa.next_state(torch.ones(1000), "Lock", "nonexistent", ["Lock"])


class TestNGramEncoder:
    def test_encode_sequence(self):
        enc = NGramEncoder(dim=4000, n=3, seed=42)
        seq = list("helloworld")
        hv = enc.encode(seq)
        assert hv.shape == (4000,)

    def test_similar_sequences(self):
        enc = NGramEncoder(dim=5000, n=3, seed=42)
        sim_similar = enc.similarity_between(
            list("helloworld"),
            list("felloworld"),
        )
        sim_different = enc.similarity_between(
            list("helloworld"),
            list("hejvarlden"),
        )
        assert sim_similar > sim_different  # Similar is more similar

    def test_different_sequences_ranked(self):
        enc = NGramEncoder(dim=5000, n=3, seed=42)
        seq_a = list("helloworld")
        seq_similar = list("felloworld")
        seq_different = list("hejvarlden")
        hv_a = enc.encode(seq_a)
        sim_similar = enc.contains_ngram(hv_a, list("ell"))
        sim_absent = enc.contains_ngram(hv_a, list("abc"))
        assert sim_similar > sim_absent  # Present n-gram ranks higher

    def test_contains_ngram(self):
        enc = NGramEncoder(dim=5000, n=3, seed=42)
        stats = enc.encode(list("helloworld"))
        present = enc.contains_ngram(stats, list("ell"))
        absent = enc.contains_ngram(stats, list("abc"))
        assert present > absent

    def test_short_sequence(self):
        enc = NGramEncoder(dim=1000, n=3)
        hv = enc.encode(list("ab"))  # shorter than n
        assert hv.sum().item() == 0


class TestFrequencyEncoder:
    def test_encode_distribution(self):
        enc = FrequencyEncoder(dim=5000, seed=42)
        dist = enc.encode({"a": 5.0, "b": 3.0, "c": 1.0})
        assert dist.shape == (5000,)

    def test_rank_symbols(self):
        enc = FrequencyEncoder(dim=5000, seed=42)
        dist = enc.encode({"a": 5.0, "b": 3.0, "c": 1.0})
        results = enc.rank_symbols(dist, ["a", "b", "c"])
        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == "c"
        assert results[0][1] >= results[1][1] >= results[2][1]

    def test_empty_distribution(self):
        enc = FrequencyEncoder(dim=1000)
        hv = enc.encode({})
        assert hv.sum().item() == 0


class TestStackEncoder:
    def test_encode_stack(self):
        s = StackEncoder(dim=5000, seed=42)
        stack = s.encode(["top", "middle", "bottom"])
        assert stack.shape == (5000,)

    def test_push(self):
        s = StackEncoder(dim=5000, seed=42)
        stack = s.encode(["b", "c"])
        new_stack = s.push(stack, "a")
        assert new_stack.shape == (5000,)
        top, _ = s.peek(new_stack, ["a", "b", "c"])
        assert top == "a"

    def test_peek(self):
        s = StackEncoder(dim=5000, seed=42)
        stack = s.encode(["top", "middle", "bottom"])
        top_name, conf = s.peek(stack, ["top", "middle", "bottom"])
        assert top_name == "top"
        assert conf > 0.5

    def test_pop(self):
        s = StackEncoder(dim=5000, seed=42)
        stack = s.encode(["a", "b", "c"])
        popped, new_stack = s.pop(stack)
        assert popped == "a"
        # After popping, "b" should be on top
        top, _ = s.peek(new_stack, ["b", "c", "a"])
        assert top == "b"

    def test_empty_stack(self):
        s = StackEncoder(dim=1000)
        stack = s.encode([])
        assert stack.sum().item() == 0