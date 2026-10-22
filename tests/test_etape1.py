# Tests des taux de l'étape 1 (comparaison avec un classifieur « toujours 0 »).
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import sys
import unittest

sys.path.insert(0, "src")
from etape1 import taux_reconnaissance, taux_toujours_zero  # noqa: E402


class TestReconnaissance(unittest.TestCase):

    def setUp(self):
        # Copies faites à la main : (classe, p, pixels) avec 2 pixels seulement
        self.copies = [
            (0, 50, [1, 0]),
            (0, 50, [0, 1]),
            (1, 50, [1, 1]),
            (1, 50, [0, 0]),
            (1, 10, [1, 1]),   # autre niveau de bruit : ne doit pas compter à 50 %
        ]

    def test_toujours_zero(self):
        # 2 zéros sur 4 copies à 50 %
        self.assertAlmostEqual(taux_toujours_zero(self.copies, 50), 50.0)
        # à 10 % il n'y a qu'un « un » : le classifieur constant se trompe toujours
        self.assertAlmostEqual(taux_toujours_zero(self.copies, 10), 0.0)

    def test_neurone_qui_repond_toujours_0(self):
        # Poids nuls et biais négatif : a = -0,5 < 0, donc ŷ = 0 partout
        self.assertAlmostEqual(taux_reconnaissance([0.0, 0.0], -0.5, self.copies, 50),
                               taux_toujours_zero(self.copies, 50))

    def test_neurone_a_deux_poids(self):
        # ŷ = 1 si les deux pixels sont allumés : a = x0 + x1 - 1,5 >= 0
        # À 50 % : juste sur les deux zéros et sur [1, 1], faux sur [0, 0] -> 75 %
        self.assertAlmostEqual(taux_reconnaissance([1.0, 1.0], -1.5, self.copies, 50), 75.0)


if __name__ == "__main__":
    unittest.main()
