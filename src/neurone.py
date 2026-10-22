# Étape 1 — un neurone : score, décision, mise à jour des poids.
#
#   score       a = w0*x0 + w1*x1 + ... + w47*x47 + b
#   décision    ŷ = 1 si a >= 0, sinon 0
#   erreur      e = d - ŷ          (d = classe attendue)
#   gradients   gj = -e * xj   ;   gb = -e
#   mise à jour wj = wj - eta * gj ;  b = b - eta * gb
#
# Le biais initial b = -0,5 correspond au seuil 0,5.

import random

from lecture import lire_motif


def initialiser_poids(nombre, graine):
    """Tire `nombre` poids au hasard entre -0,1 et 0,1.

    Avec la même graine, on obtient toujours les mêmes poids.
    """
    generateur = random.Random(graine)
    poids = []
    for _ in range(nombre):
        poids.append(generateur.uniform(-0.1, 0.1))
    return poids


def calculer_score(poids, biais, pixels):
    """a = somme des poids[j] * pixels[j], plus le biais."""
    a = 0.0
    for j in range(len(pixels)):
        a = a + poids[j] * pixels[j]
    a = a + biais
    return a


def decider(a):
    """ŷ = 1 si a >= 0, sinon 0."""
    if a >= 0:
        return 1
    else:
        return 0


def mettre_a_jour(poids, biais, pixels, d, eta):
    """Une correction (règle du perceptron) sur un exemple.

    Renvoie les nouveaux poids, le nouveau biais et l'erreur e.
    Si la décision est déjà juste, e = 0 et rien ne change.
    """
    a = calculer_score(poids, biais, pixels)
    y = decider(a)
    e = d - y

    nouveaux_poids = []
    for j in range(len(poids)):
        g = -e * pixels[j]
        nouveaux_poids.append(poids[j] - eta * g)

    gb = -e
    nouveau_biais = biais - eta * gb

    return nouveaux_poids, nouveau_biais, e


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python3 src/neurone.py

    # 1) Calcul sur papier du sujet (2 entrées seulement)
    x = [1, 0]
    w = [0.2, -0.1]
    b = -0.5
    d = 1
    eta = 0.01

    a = calculer_score(w, b, x)
    y = decider(a)
    print("Calcul sur papier")
    print(f"  a = {a:.4f}")
    print(f"  ŷ = {y}")
    print(f"  e = {d - y}")

    w, b, e = mettre_a_jour(w, b, x, d, eta)
    print(f"  après la mise à jour : w0 = {w[0]:.4f}, w1 = {w[1]:.4f}, b = {b:.4f}")

    # 2) Les vrais motifs, avec des poids tirés au hasard (graine 42)
    poids = initialiser_poids(48, 42)
    biais = -0.5
    print()
    print("Motifs avec les poids initiaux (graine 42, b = -0.5)")
    for chemin in ["data/zero.txt", "data/un.txt"]:
        pixels, classe = lire_motif(chemin)
        a = calculer_score(poids, biais, pixels)
        print(f"  {chemin} : classe {classe}, a = {a:.4f}, ŷ = {decider(a)}")
