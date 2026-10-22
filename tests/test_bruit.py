# Tests des copies bruitées (étape 1, activité 4).
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import random
import sys
import unittest

sys.path.insert(0, "src")
from bruit import arrondi, bruiter, generer_copies, nombre_pixels_a_inverser  # noqa: E402
from lecture import lire_motif  # noqa: E402


class TestBruit(unittest.TestCase):

    def test_arrondi(self):
        self.assertEqual(arrondi(2.5), 3)    # round(2.5) donnerait 2
        self.assertEqual(arrondi(-2.5), -3)
        self.assertEqual(arrondi(4.8), 5)
        self.assertEqual(arrondi(14.4), 14)

    def test_nombre_de_pixels(self):
        attendu = {0: 0, 10: 5, 20: 10, 30: 14, 40: 19, 50: 24,
                   60: 29, 70: 34, 80: 38, 90: 43, 100: 48}
        for p in attendu:
            self.assertEqual(nombre_pixels_a_inverser(p), attendu[p])

    def test_exactement_n_pixels_distincts(self):
        pixels, classe = lire_motif("data/zero.txt")
        original = list(pixels)
        generateur = random.Random(0)
        for n in range(49):
            copie = bruiter(pixels, n, generateur)
            differences = 0
            for j in range(48):
                if copie[j] != pixels[j]:
                    differences = differences + 1
            self.assertEqual(differences, n)
        self.assertEqual(pixels, original)  # le motif d'origine n'a pas changé

    def test_100_pourcent_donne_le_negatif(self):
        pixels, classe = lire_motif("data/un.txt")
        copie = bruiter(pixels, 48, random.Random(1))
        for j in range(48):
            self.assertEqual(copie[j], 1 - pixels[j])

    def test_meme_graine_memes_copies(self):
        motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]
        a = generer_copies(motifs, [0, 10, 50], 5, 2026)
        b = generer_copies(motifs, [0, 10, 50], 5, 2026)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2 * 3 * 5)  # 2 chiffres × 3 niveaux × 5 copies


if __name__ == "__main__":
    unittest.main()
