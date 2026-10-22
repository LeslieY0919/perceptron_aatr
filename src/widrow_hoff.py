# Étape 2 — règle de Widrow–Hoff.
#
# Le score est le même qu'à l'étape 1 : a = Σ wj xj + b.
# Mais la correction utilise la valeur du score, pas seulement la décision :
#   delta = a - d
#   wj = wj - eta * delta * xj
#   b  = b  - eta * delta
# Une décision déjà juste continue donc à rapprocher le score de 0 ou de 1.
# Pour classer : prédiction = 1 si a >= 0,5, sinon 0.
# Perte sur N motifs (poids figés) : L = (1 / 2N) * Σ (dk - ak)²

import math
import random

from lecture import lire_motif
from neurone import calculer_score, mettre_a_jour

SEUIL = 0.5


def predire(score):
    """Prédiction = 1 si le score >= 0,5, sinon 0."""
    if score >= SEUIL:
        return 1
    else:
        return 0


def mettre_a_jour_wh(poids, biais, pixels, target, eta):
    """Une correction de Widrow–Hoff. Renvoie (poids, biais, delta)."""
    score = calculer_score(poids, biais, pixels)
    delta = score - target

    nouveaux_poids = []
    for j in range(len(poids)):
        nouveaux_poids.append(poids[j] - eta * delta * pixels[j])
    nouveau_biais = biais - eta * delta

    return nouveaux_poids, nouveau_biais, delta


def perte(poids, biais, exemples):
    """L = (1 / 2N) * Σ (d - a)², calculée avec des poids figés."""
    somme = 0.0
    for pixels, target in exemples:
        score = calculer_score(poids, biais, pixels)
        ecart = target - score
        somme = somme + ecart * ecart  # ecart ** 2 planterait si le score explose
    return somme / (2 * len(exemples))


def entrainer_wh(poids, biais, exemples, eta, max_epoques, graine_ordre, perte_cible=1e-4):
    """Apprend avec Widrow–Hoff, une époque = chaque motif une fois, ordre mélangé.

    Renvoie (poids, biais, pertes, statut).
    pertes[0] = perte avec les poids initiaux.
    statut : "perte atteinte", "limite d'époques" ou "perte non finie".
    """
    generateur = random.Random(graine_ordre)
    ordre = list(range(len(exemples)))
    pertes = [perte(poids, biais, exemples)]

    for epoque in range(1, max_epoques + 1):
        generateur.shuffle(ordre)
        for k in ordre:
            pixels, target = exemples[k]
            poids, biais, delta = mettre_a_jour_wh(poids, biais, pixels, target, eta)

        L = perte(poids, biais, exemples)
        pertes.append(L)
        if not math.isfinite(L):
            return poids, biais, pertes, "perte non finie"
        if L <= perte_cible:
            return poids, biais, pertes, "perte atteinte"

    return poids, biais, pertes, "limite d'époques"


def stabilite(pixels, eta):
    """eta * ||[x, 1]||² : somme des carrés des pixels, plus 1 pour le biais.

    Pour une présentation, il faut que cette valeur reste < 2.
    """
    somme = 1  # l'entrée du biais vaut toujours 1
    for x in pixels:
        somme = somme + x * x
    return eta * somme


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python3 src/widrow_hoff.py

    # 1) Même petit exemple qu'à l'étape 1 : x = [1, 0], w = [0,2 ; -0,1], b = -0,5, d = 1
    x = [1, 0]
    w = [0.2, -0.1]
    b = -0.5
    target = 1
    eta = 0.01
    score = calculer_score(w, b, x)
    print("Petit exemple (calcul sur papier)")
    print(f"  score a = {score:.4f}")

    w_p, b_p, e = mettre_a_jour(w, b, x, target, eta)
    print(f"  perceptron  : e = d - ŷ = {e}, w0 = {w_p[0]:.4f}, b = {b_p:.4f}")

    w_wh, b_wh, delta = mettre_a_jour_wh(w, b, x, target, eta)
    print(f"  Widrow–Hoff : delta = a - d = {delta:.4f}, w0 = {w_wh[0]:.4f}, b = {b_wh:.4f}")

    # 2) Classe déjà juste mais score 0,6 au lieu de 1 (fiche, étape 2, question 1)
    print()
    print("Score 0,6 pour la cible 1 (un seul pixel éteint, seul le biais compte)")
    w_p, b_p, e = mettre_a_jour([0.0], 0.6, [0], 1, eta)
    w_wh, b_wh, delta = mettre_a_jour_wh([0.0], 0.6, [0], 1, eta)
    print(f"  prédiction = {predire(0.6)}")
    print(f"  perceptron  : e = {e}, b devient {b_p:.4f}")
    print(f"  Widrow–Hoff : delta = {delta:.4f}, b devient {b_wh:.4f}")

    # 3) Contrôle de stabilité sur les vrais motifs (fiche, étape 2, question 3)
    print()
    print(f"Stabilité : eta * ||[x,1]||² doit rester < 2 (eta = {eta})")
    for chemin in ["data/zero.txt", "data/un.txt"]:
        pixels, classe = lire_motif(chemin)
        print(f"  {chemin} : {stabilite(pixels, eta):.3f}")
