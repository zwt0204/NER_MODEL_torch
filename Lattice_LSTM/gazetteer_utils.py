# -*- encoding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class TrieNode:
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: Sequence[str]):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    def search(self, word: Sequence[str]) -> bool:
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def enumerate_match(self, word: List[str]) -> List[str]:
        matched = []
        probe = list(word)
        while len(probe) > 1:
            if self.search(probe):
                matched.append(''.join(probe))
            del probe[-1]
        return matched


class Gazetteer:
    def __init__(self, lower: bool = False):
        self.trie = Trie()
        self.ent2id = {'<UNK>': 0}
        self.lower = lower

    def _normalize(self, word_list: Sequence[str]) -> List[str]:
        if self.lower:
            return [word.lower() for word in word_list]
        return list(word_list)

    def insert(self, word_list: Sequence[str]):
        tokens = self._normalize(word_list)
        if len(tokens) <= 1:
            return
        self.trie.insert(tokens)
        key = ''.join(tokens)
        if key not in self.ent2id:
            self.ent2id[key] = len(self.ent2id)

    def enumerate_match_list(self, word_list: Sequence[str]) -> List[str]:
        tokens = self._normalize(word_list)
        return self.trie.enumerate_match(tokens)

    def search_id(self, word_list: Sequence[str]) -> int:
        tokens = self._normalize(word_list)
        return self.ent2id.get(''.join(tokens), 0)

    def size(self) -> int:
        return len(self.ent2id)


def load_gazetteer(gazetteer_file: str, lower: bool = False) -> Gazetteer:
    gaz = Gazetteer(lower=lower)
    with open(gazetteer_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            term = line.strip()
            if term:
                gaz.insert(list(term))
    return gaz
