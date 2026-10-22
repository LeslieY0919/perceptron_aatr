# Étape 1 — expérience complète : apprendre 0 et 1, puis tester le bruit.
# Lancer depuis la racine du dépôt : python3 src/etape1.py
# Les résultats (CSV, poids, figure) sont écrits dans resultats/.

from apprentissage import entrainer
from bruit import generer_copies, nombre_pixels_a_inverser
from fichiers import ecrire_csv, ecrire_json
from lecture import lire_motif
from neurone import calculer_score, decider, initialiser_poids

# Réglages : en changer un seul à la fois, relancer, comparer.
GRAINE_POIDS = 42      # poids initiaux
GRAINE_ORDRE = 1       # ordre des motifs à chaque époque
GRAINE_BRUIT = 2026    # copies bruitées
BIAIS_INITIAL = -0.5
ETA = 0.01
MAX_EPOQUES = 1000
NIVEAUX_BRUIT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # p en %
NB_COPIES = 50         # copies par chiffre et par niveau
AFFICHER_EPOQUES = True


def taux_erreur_bruit(poids, biais, copies, classe, p):
    """Taux d'erreur (%) sur les copies d'un chiffre à un niveau de bruit p.

    Taux = 100 × nombre de mauvaises réponses / nombre d'essais. Poids figés.
    """
    essais = 0
    erreurs = 0
    for c, niveau, pixels in copies:
        if c == classe and niveau == p:
            essais = essais + 1
            if decider(calculer_score(poids, biais, pixels)) != classe:
                erreurs = erreurs + 1
    return 100 * erreurs / essais


def taux_reconnaissance(poids, biais, copies, p):
    """Bonnes réponses (%) sur les copies des DEUX chiffres au niveau de bruit p.

    Taux = 100 × nombre de bonnes réponses / nombre d'essais. Poids figés.
    """
    essais = 0
    bonnes = 0
    for classe, niveau, pixels in copies:
        if niveau == p:
            essais = essais + 1
            if decider(calculer_score(poids, biais, pixels)) == classe:
                bonnes = bonnes + 1
    return 100 * bonnes / essais


def taux_toujours_zero(copies, p):
    """Même mesure pour un classifieur qui répond toujours 0, sans rien apprendre.

    Sert de point de comparaison (fiche, étape 1, question 4 : prévoir avant de lancer).
    """
    essais = 0
    bonnes = 0
    for classe, niveau, pixels in copies:
        if niveau == p:
            essais = essais + 1
            if classe == 0:   # il répond 0 : il a raison seulement sur les zéros
                bonnes = bonnes + 1
    return 100 * bonnes / essais


def tracer_figure(erreurs_par_epoque, taux, chemin):
    """Figure à deux panneaux : erreurs selon l'époque ; taux d'erreur selon le bruit."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib absent : pas de figure (pip install matplotlib)")
        return

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(11, 4))

    epoques = list(range(len(erreurs_par_epoque)))
    gauche.plot(epoques, erreurs_par_epoque, "o-")
    gauche.set_xlabel("Époque (0 = poids initiaux)")
    gauche.set_ylabel("Erreurs sur les 2 motifs")
    gauche.set_yticks([0, 1, 2])
    gauche.set_title("Apprentissage (règle du perceptron)")
    gauche.grid(alpha=0.3)

    for classe, style in [(0, "o-"), (1, "s--")]:
        valeurs = [taux[(classe, p)] for p in NIVEAUX_BRUIT]
        droite.plot(NIVEAUX_BRUIT, valeurs, style, label=f"chiffre {classe}")
    droite.set_xlabel("Bruit p (% des 48 pixels inversés)")
    droite.set_ylabel("Taux d'erreur (%)")
    droite.set_title(f"Robustesse au bruit ({NB_COPIES} copies par point)")
    droite.legend()
    droite.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(chemin, dpi=120)
    print(f"  écrit : {chemin}")


def main():
    # Activité 1 : les deux motifs et les poids initiaux
    motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]
    poids_initiaux = initialiser_poids(48, GRAINE_POIDS)
    print(f"Graine des poids {GRAINE_POIDS}, b = {BIAIS_INITIAL}, eta = {ETA}")

    # Activités 2 et 3 : apprendre
    poids, biais, erreurs_par_epoque = entrainer(
        poids_initiaux, BIAIS_INITIAL, motifs, ETA, MAX_EPOQUES, GRAINE_ORDRE, AFFICHER_EPOQUES
    )
    nb_epoques = len(erreurs_par_epoque) - 1
    if erreurs_par_epoque[-1] == 0:
        print(f"Zéro erreur après {nb_epoques} époque(s).")
    else:
        print(f"ATTENTION : limite de {MAX_EPOQUES} époques atteinte, "
              f"{erreurs_par_epoque[-1]} erreur(s) restante(s).")
    for pixels, classe in motifs:
        a = calculer_score(poids, biais, pixels)
        print(f"  chiffre {classe} : a = {a:+.4f}, ŷ = {decider(a)}")

    # Activité 4 : copies bruitées (jamais apprises)
    copies = generer_copies(motifs, NIVEAUX_BRUIT, NB_COPIES, GRAINE_BRUIT)
    taux = {}  # taux[(chiffre, p)] = taux d'erreur en %
    print()
    print("  p (%) | pixels inversés | erreur chiffre 0 | erreur chiffre 1")
    for p in NIVEAUX_BRUIT:
        for classe in [0, 1]:
            taux[(classe, p)] = taux_erreur_bruit(poids, biais, copies, classe, p)
        print(f"  {p:5d} | {nombre_pixels_a_inverser(p):15d} | "
              f"{taux[(0, p)]:14.1f} % | {taux[(1, p)]:14.1f} %")

    # Comparaison : notre neurone face à un classifieur « toujours 0 »
    reconnaissance = []
    print()
    print("  p (%) | reconnaissance du neurone | classifieur « toujours 0 »")
    for p in NIVEAUX_BRUIT:
        neurone = taux_reconnaissance(poids, biais, copies, p)
        constant = taux_toujours_zero(copies, p)
        reconnaissance.append([p, 2 * NB_COPIES, neurone, constant])
        print(f"  {p:5d} | {neurone:23.1f} % | {constant:24.1f} %")

    # Résultats à conserver
    print()
    lignes = []
    for epoque in range(len(erreurs_par_epoque)):
        lignes.append([epoque, erreurs_par_epoque[epoque]])
    ecrire_csv("resultats/etape1_apprentissage.csv", ["epoque", "erreurs"], lignes)

    lignes = []
    for classe in [0, 1]:
        for p in NIVEAUX_BRUIT:
            lignes.append([classe, p, nombre_pixels_a_inverser(p), NB_COPIES, taux[(classe, p)]])
    ecrire_csv("resultats/etape1_bruit.csv",
               ["chiffre", "bruit_pourcent", "pixels_inverses", "essais", "taux_erreur_pourcent"],
               lignes)
    ecrire_csv("resultats/etape1_reconnaissance.csv",
               ["bruit_pourcent", "essais", "reconnaissance_neurone_pourcent",
                "reconnaissance_toujours_0_pourcent"], reconnaissance)

    lignes = []
    for classe, p, pixels in copies:
        lignes.append([classe, p, "".join(str(x) for x in pixels)])
    ecrire_csv("resultats/copies_bruitees.csv", ["chiffre", "bruit_pourcent", "pixels"], lignes)

    ecrire_json("resultats/etape1_poids.json", {
        "graine_poids": GRAINE_POIDS, "graine_ordre": GRAINE_ORDRE, "graine_bruit": GRAINE_BRUIT,
        "eta": ETA, "biais_initial": BIAIS_INITIAL, "epoques": nb_epoques,
        "poids_initiaux": poids_initiaux, "poids_appris": poids, "biais_appris": biais,
    })
    tracer_figure(erreurs_par_epoque, taux, "resultats/etape1.png")


if __name__ == "__main__":
    main()
