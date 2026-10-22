# Tests de la règle de Widrow–Hoff (étape 2).
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import sys
import unittest

sys.path.insert(0, "src")
from lecture import lire_motif  # noqa: E402
from neurone import initialiser_poids, mettre_a_jour  # noqa: E402
from widrow_hoff import entrainer_wh, mettre_a_jour_wh, perte, predire, stabilite  # noqa: E402


class TestPetitExemple(unittest.TestCase):
    # Même exemple que sur papier : x = [1, 0], w = [0,2 ; -0,1], b = -0,5, d = 1

    def test_widrow_hoff(self):
        w, b, delta = mettre_a_jour_wh([0.2, -0.1], -0.5, [1, 0], 1, 0.01)
        self.assertAlmostEqual(delta, -1.3)    # a - d = -0,3 - 1
        self.assertAlmostEqual(w[0], 0.213)    # 0,2 - 0,01 * (-1,3) * 1
        self.assertAlmostEqual(w[1], -0.1)
        self.assertAlmostEqual(b, -0.487)

    def test_comparaison_avec_le_perceptron(self):
        # Perceptron : e = 1 -> w0 = 0,21 ; Widrow–Hoff : delta = -1,3 -> w0 = 0,213
        w_p, b_p, e = mettre_a_jour([0.2, -0.1], -0.5, [1, 0], 1, 0.01)
        w_wh, b_wh, delta = mettre_a_jour_wh([0.2, -0.1], -0.5, [1, 0], 1, 0.01)
        self.assertAlmostEqual(w_p[0], 0.21)
        self.assertGreater(w_wh[0], w_p[0])

    def test_la_correction_continue_si_la_classe_est_juste(self):
        # Score 0,6 pour la cible 1 : classe juste (0,6 >= 0,5), mais delta = -0,4
        w, b, delta = mettre_a_jour_wh([0.0], 0.6, [0], 1, 0.01)
        self.assertEqual(predire(0.6), 1)
        self.assertAlmostEqual(delta, -0.4)
        self.assertNotEqual(b, 0.6)


class TestPerteEtStabilite(unittest.TestCase):

    def test_perte(self):
        # Deux exemples, scores 0,5 et 0 pour des cibles 1 et 0 : L = (0,25 + 0) / 4
        exemples = [([1], 1), ([0], 0)]
        self.assertAlmostEqual(perte([0.5], 0.0, exemples), 0.0625)

    def test_stabilite_des_motifs(self):
        zero, c0 = lire_motif("data/zero.txt")
        un, c1 = lire_motif("data/un.txt")
        self.assertAlmostEqual(stabilite(zero, 0.01), 0.17)  # 16 pixels + 1
        self.assertAlmostEqual(stabilite(un, 0.01), 0.10)    # 9 pixels + 1

    def test_seuil(self):
        self.assertEqual(predire(0.5), 1)
        self.assertEqual(predire(0.49), 0)


class TestApprentissageWH(unittest.TestCase):

    def test_la_perte_diminue(self):
        motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]
        poids = initialiser_poids(48, 42)
        poids, biais, pertes, statut = entrainer_wh(poids, -0.5, motifs, 0.01, 1000, 1)
        self.assertLess(pertes[-1], pertes[0])
        self.assertIn(statut, ["perte atteinte", "limite d'époques"])

    def test_pas_trop_grand_detecte(self):
        # eta * ||[x,1]||² = 1 * 17 >> 2 : la perte explose
        motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]
        poids, biais, pertes, statut = entrainer_wh([0.0] * 48, 0.0, motifs, 1.0, 1000, 1)
        self.assertEqual(statut, "perte non finie")


if __name__ == "__main__":
    unittest.main()
