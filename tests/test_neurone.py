# Tests du neurone (score, décision, mise à jour).
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import sys
import unittest

sys.path.insert(0, "src")
from neurone import calculer_score, decider, initialiser_poids, mettre_a_jour  # noqa: E402


class TestCalculSurPapier(unittest.TestCase):
    # x = [1, 0], w = [0,2 ; -0,1], b = -0,5, d = 1, eta = 0,01

    def test_score_et_decision(self):
        a = calculer_score([0.2, -0.1], -0.5, [1, 0])
        self.assertAlmostEqual(a, -0.3)   # 0,2*1 + (-0,1)*0 - 0,5
        self.assertEqual(decider(a), 0)   # a < 0

    def test_mise_a_jour(self):
        w, b, e = mettre_a_jour([0.2, -0.1], -0.5, [1, 0], 1, 0.01)
        self.assertEqual(e, 1)            # e = 1 - 0
        self.assertAlmostEqual(w[0], 0.21)
        self.assertAlmostEqual(w[1], -0.1)  # x1 = 0 : ce poids ne bouge pas
        self.assertAlmostEqual(b, -0.49)


class TestNeurone(unittest.TestCase):

    def test_rien_ne_change_si_la_decision_est_juste(self):
        w, b, e = mettre_a_jour([0.2, -0.1], -0.5, [1, 0], 0, 0.01)
        self.assertEqual(e, 0)
        self.assertEqual(w, [0.2, -0.1])
        self.assertEqual(b, -0.5)

    def test_seuil(self):
        self.assertEqual(decider(0.0), 1)   # a = 0 compte comme 1
        self.assertEqual(decider(-0.001), 0)

    def test_poids_initiaux(self):
        poids = initialiser_poids(48, 42)
        self.assertEqual(len(poids), 48)
        for p in poids:
            self.assertTrue(-0.1 <= p <= 0.1)
        self.assertEqual(poids, initialiser_poids(48, 42))  # même graine, mêmes poids


if __name__ == "__main__":
    unittest.main()
