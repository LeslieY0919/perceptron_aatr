# Tests de l'apprentissage par époques (étape 1, activité 3).
# Lancer depuis la racine du dépôt : python3 -m unittest discover -s tests -v

import sys
import unittest

sys.path.insert(0, "src")
from apprentissage import compter_erreurs, entrainer  # noqa: E402
from lecture import lire_motif  # noqa: E402
from neurone import initialiser_poids  # noqa: E402


class TestApprentissage(unittest.TestCase):

    def setUp(self):
        self.motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]

    def test_zero_erreur_a_la_fin(self):
        poids = initialiser_poids(48, 42)
        poids, biais, erreurs = entrainer(poids, -0.5, self.motifs, 0.01, 1000, 1)
        self.assertEqual(erreurs[-1], 0)
        self.assertEqual(compter_erreurs(poids, biais, self.motifs), 0)

    def test_s_arrete_a_la_premiere_epoque_sans_erreur(self):
        poids = initialiser_poids(48, 42)
        poids, biais, erreurs = entrainer(poids, -0.5, self.motifs, 0.01, 1000, 1)
        for e in erreurs[1:-1]:
            self.assertGreater(e, 0)

    def test_limite_d_epoques(self):
        # Le même motif avec deux classes : impossible à séparer.
        pixels = self.motifs[0][0]
        exemples = [(pixels, 0), (pixels, 1)]
        poids, biais, erreurs = entrainer([0.0] * 48, -0.5, exemples, 0.01, 20, 1)
        self.assertEqual(len(erreurs), 21)  # époque 0 + 20 époques
        self.assertEqual(erreurs[-1], 1)

    def test_compter_ne_modifie_pas_les_poids(self):
        poids = initialiser_poids(48, 42)
        copie = list(poids)
        compter_erreurs(poids, -0.5, self.motifs)
        self.assertEqual(poids, copie)


if __name__ == "__main__":
    unittest.main()
