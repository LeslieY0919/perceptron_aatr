# Étape 1, activité 4 — copies bruitées des motifs.
#
# Pour p % de bruit, on inverse exactement n = arrondi(48 * p / 100) pixels
# DISTINCTS, tirés au hasard (0 devient 1, 1 devient 0).
# Ces copies servent seulement à tester : on n'apprend jamais dessus.

import math
import random


def arrondi(v):
    """Arrondi au plus proche ; à mi-distance, on s'éloigne de zéro (2,5 -> 3).

    On n'utilise pas round() : Python arrondit au pair (round(2.5) donne 2).
    """
    if v >= 0:
        return math.floor(v + 0.5)
    else:
        return -math.floor(-v + 0.5)


def nombre_pixels_a_inverser(p, nb_pixels=48):
    """n = arrondi(nb_pixels * p / 100)."""
    return arrondi(nb_pixels * p / 100)


def bruiter(pixels, n, generateur):
    """Copie du motif où n pixels distincts sont inversés.

    Le motif d'origine n'est pas modifié.
    """
    copie = list(pixels)
    positions = generateur.sample(range(len(pixels)), n)  # n positions toutes différentes
    for j in positions:
        copie[j] = 1 - copie[j]
    return copie


def generer_copies(motifs, niveaux, nb_copies, graine):
    """Copies bruitées de chaque motif, pour chaque niveau de bruit.

    motifs : liste de (pixels, classe) ; niveaux : liste de p en %.
    Renvoie une liste de (classe, p, copie).
    Même graine -> exactement les mêmes copies (l'étape 2 s'en sert).
    """
    generateur = random.Random(graine)
    copies = []
    for pixels, classe in motifs:
        for p in niveaux:
            n = nombre_pixels_a_inverser(p, len(pixels))
            for _ in range(nb_copies):
                copies.append((classe, p, bruiter(pixels, n, generateur)))
    return copies
