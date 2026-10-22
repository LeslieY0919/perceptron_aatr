# Étape 1, activité 3 — apprentissage par époques (règle du perceptron).
#
# Une époque présente chaque motif une fois, dans un ordre mélangé.
# À la fin de l'époque, on recompte les erreurs SANS modifier les poids.
# On s'arrête dès qu'il n'y a plus d'erreur, ou après max_epoques.

import random

from neurone import calculer_score, decider, mettre_a_jour


def compter_erreurs(poids, biais, exemples):
    """Nombre de motifs mal classés. Les poids ne sont pas modifiés."""
    erreurs = 0
    for pixels, d in exemples:
        a = calculer_score(poids, biais, pixels)
        if decider(a) != d:
            erreurs = erreurs + 1
    return erreurs


def entrainer(poids, biais, exemples, eta, max_epoques, graine_ordre, afficher=False):
    """Apprend avec la règle du perceptron.

    exemples : liste de (pixels, classe).
    Renvoie (poids, biais, erreurs_par_epoque).
    erreurs_par_epoque[0] = erreurs avec les poids initiaux, avant apprentissage.
    """
    generateur = random.Random(graine_ordre)
    ordre = list(range(len(exemples)))
    erreurs_par_epoque = [compter_erreurs(poids, biais, exemples)]

    for epoque in range(1, max_epoques + 1):
        generateur.shuffle(ordre)
        for k in ordre:
            pixels, d = exemples[k]
            poids, biais, e = mettre_a_jour(poids, biais, pixels, d, eta)

        erreurs = compter_erreurs(poids, biais, exemples)
        erreurs_par_epoque.append(erreurs)
        if afficher:
            print(f"  époque {epoque} : ordre {ordre}, erreurs = {erreurs}, b = {biais:.4f}")

        if erreurs == 0:
            break

    return poids, biais, erreurs_par_epoque
