# Tests de la lecture des motifs.
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import sys
import unittest

sys.path.insert(0, "src")
from lecture import lire_motif  # noqa: E402


class TestLecture(unittest.TestCase):

    def test_zero(self):
        pixels, classe = lire_motif("data/zero.txt")
        self.assertEqual(len(pixels), 48)
        self.assertEqual(classe, 0)
        self.assertEqual(sum(pixels), 16)
        self.assertEqual(pixels[0:6], [0, 0, 0, 0, 0, 0])   # 1re ligne : ......
        self.assertEqual(pixels[6:12], [0, 1, 1, 1, 1, 0])  # 2e ligne : .****.

    def test_un(self):
        pixels, classe = lire_motif("data/un.txt")
        self.assertEqual(len(pixels), 48)
        self.assertEqual(classe, 1)
        self.assertEqual(sum(pixels), 9)
        self.assertEqual(pixels[6:12], [0, 0, 0, 1, 0, 0])  # 2e ligne : ...*..


if __name__ == "__main__":
    unittest.main()
