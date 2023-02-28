from typing import List
from histr import Shabdansh
import logging

logger = logging.getLogger(__name__)


class GraphemeVocab:
    def __init__(self):
        self.stoi = None
        self.itos = None

    def build_vocab(self, texts: List[str]) -> None:
        all_grapheme_clusters = set()
        for text in texts:
            grapheme_clusters = list(Shabdansh(text))
            all_grapheme_clusters.update(grapheme_clusters)
        self.stoi = {grapheme: i for i, grapheme in enumerate(all_grapheme_clusters)}
        self.itos = {i: grapheme for grapheme, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        text = Shabdansh(text)
        ids = []

        # check vocab is already present
        if not self.stoi:
            raise Exception("Please call build_vocab before calling encode")

        try:
            # get the ids
            ids = [self.stoi[grapheme] for grapheme in text]
        except KeyError:
            logger.error("input contains token which were not used in `build_vocab`", exc_info=True)

        return ids

    def decode(self, ids: List[int]) -> str:
        if not self.itos:
            raise Exception("Please call build_vocab before calling decode")

        return "".join([self.itos[id] for id in ids])
