# Étape 1 — lecture d'un motif.
# Un fichier motif contient 8 lignes de 6 caractères, puis la classe sur la
# 9e ligne. '.' devient 0 et '*' devient 1. On lit ligne par ligne, de gauche
# à droite : on obtient 48 entrées x0 ... x47.


def lire_motif(chemin):
    """Renvoie (pixels, classe) : une liste de 48 entiers 0/1 et la classe."""
    with open(chemin, encoding="utf-8") as fichier:
        lignes = fichier.read().split()

    pixels = []
    for ligne in lignes[:8]:  # les 8 lignes de pixels
        for caractere in ligne:
            if caractere == "*":
                pixels.append(1)
            elif caractere == ".":
                pixels.append(0)
            else:
                raise ValueError(f"caractère inconnu : {caractere!r}")

    if len(pixels) != 48:
        raise ValueError(f"{len(pixels)} pixels lus au lieu de 48")

    classe = int(lignes[8])  # la 9e ligne
    return pixels, classe


def afficher_motif(pixels):
    """Affiche les 48 pixels en grille 8 × 6, pour vérifier à l'œil."""
    for i in range(8):
        ligne = ""
        for j in range(6):
            if pixels[i * 6 + j] == 1:
                ligne += "*"
            else:
                ligne += "."
        print(ligne)


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python3 src/lecture.py
    for chemin in ["data/zero.txt", "data/un.txt"]:
        pixels, classe = lire_motif(chemin)
        print(f"{chemin} : classe {classe}, {sum(pixels)} pixels allumés sur {len(pixels)}")
        afficher_motif(pixels)
        print(pixels)
        print()
