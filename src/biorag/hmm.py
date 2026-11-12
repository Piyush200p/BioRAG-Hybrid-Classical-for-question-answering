"""Normalized, fixed-parameter discrete HMM and log-space Viterbi decoding.

SUPPORT is a candidate relevance-state name, not an entailment guarantee.
Parameters are hand specified for a prototype; no EM/labelled HMM training
has been performed. Sequence boundaries reset for each source document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .text import tokenize


@dataclass
class EvidenceSentence:
    sentence: str
    state: str
    score: float
    position: int


class EvidenceHMM:
    states = ("SUPPORT", "CONTEXT", "NOISE")
    observations = ("LOW", "MID", "HIGH")

    def __init__(self, support_threshold: float = 0.18, context_threshold: float = 0.06):
        if not 0 <= context_threshold < support_threshold <= 1:
            raise ValueError("Require 0 <= context threshold < support threshold <= 1")
        self.support_threshold = support_threshold
        self.context_threshold = context_threshold
        self.start = {"SUPPORT": 0.2, "CONTEXT": 0.3, "NOISE": 0.5}
        self.transition = {
            "SUPPORT": {"SUPPORT": 0.50, "CONTEXT": 0.25, "NOISE": 0.25},
            "CONTEXT": {"SUPPORT": 0.30, "CONTEXT": 0.45, "NOISE": 0.25},
            "NOISE": {"SUPPORT": 0.10, "CONTEXT": 0.15, "NOISE": 0.75},
        }
        self.emission = {
            "SUPPORT": {"LOW": 0.05, "MID": 0.20, "HIGH": 0.75},
            "CONTEXT": {"LOW": 0.20, "MID": 0.55, "HIGH": 0.25},
            "NOISE": {"LOW": 0.80, "MID": 0.15, "HIGH": 0.05},
        }

    @staticmethod
    def overlap(query: str, sentence: str) -> float:
        terms = set(tokenize(query))
        return len(terms & set(tokenize(sentence))) / len(terms) if terms else 0.0

    def observe(self, score: float) -> str:
        if score >= self.support_threshold:
            return "HIGH"
        if score >= self.context_threshold:
            return "MID"
        return "LOW"

    def decode(self, observations: list[str]) -> list[str]:
        if not observations:
            return []
        if any(item not in self.observations for item in observations):
            raise ValueError("Unknown HMM observation")
        dp = [{state: math.log(self.start[state]) + math.log(self.emission[state][observations[0]]) for state in self.states}]
        back: list[dict[str, str]] = [{}]
        for observation in observations[1:]:
            row: dict[str, float] = {}
            pointers: dict[str, str] = {}
            for state in self.states:
                previous = max(self.states, key=lambda item: dp[-1][item] + math.log(self.transition[item][state]))
                row[state] = dp[-1][previous] + math.log(self.transition[previous][state]) + math.log(self.emission[state][observation])
                pointers[state] = previous
            dp.append(row)
            back.append(pointers)
        path = [max(self.states, key=lambda state: dp[-1][state])]
        for position in range(len(observations) - 1, 0, -1):
            path.append(back[position][path[-1]])
        return list(reversed(path))

    def select(self, query: str, sentences: list[str], retrieval_score: float = 0.0) -> list[EvidenceSentence]:
        # retrieval_score retained for interface compatibility; overlap emissions
        # avoid attributing the top document's score to unrelated source sentences.
        signals = [self.overlap(query, sentence) for sentence in sentences]
        path = self.decode([self.observe(signal) for signal in signals])
        return [EvidenceSentence(sentence, label, signal, index)
                for index, (sentence, label, signal) in enumerate(zip(sentences, path, signals))
                if label != "NOISE" and signal >= self.context_threshold]
