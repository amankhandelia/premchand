import unittest
from mingpt.train import GraphemeVocab


class TestGraphemeVocab(unittest.TestCase):
    texts = [
        "साठे के जिन्दादिल नौजवानों ने रंग-बिरंगे जांघिये बनवाये",
        "साठे और पाठे दो लगे हुए मौजे थे",
        "दोनों गंगा के किनारे",
        "उन लोगन के जनम नसाये जिन पाठे मान लीन अवतार",
    ]

    def test_build_vocab(self):
        vocab = GraphemeVocab()
        vocab.build_vocab(self.texts)
        self.assertEqual(len(vocab.stoi), 45)
        self.assertEqual(len(vocab.itos), 45)

    def test_encode(self):
        vocab = GraphemeVocab()
        vocab.build_vocab(self.texts)
        encoded_text = vocab.encode("उन लोगन")
        self.assertEqual(encoded_text, [20, 32, 19, 34, 35, 32])

    def test_decode(self):
        vocab = GraphemeVocab()
        vocab.build_vocab(self.texts)
        decoded_text = vocab.decode([0, 1, 2, 2, 3, 4, 5, 6, 2, 7, 4, 8])
        self.assertEqual(decoded_text, "hello world")
