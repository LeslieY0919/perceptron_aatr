# Étape 2 — comparer la règle du perceptron et Widrow–Hoff.
# Lancer depuis la racine du dépôt : python3 src/etape2.py
#
# On garde tout comme à l'étape 1 (mêmes poids initiaux, même ordre, même eta)
# et on change une seule chose : la règle d'erreur.
#
# Fait dans cette version (activités 1 à 3) :
#   - contrôle de stabilité eta * ||[x,1]||² sur les deux motifs ;
#   - apprentissage avec les deux règles, à partir des MÊMES poids initiaux ;
#   - perte L après chaque époque, poids figés -> resultats/etape2_perte.csv ;
#   - scores et décisions des deux méthodes sur les motifs propres.
#
# À FAIRE (prochains commits) :
#   - activité 4 : tester les MÊMES copies bruitées qu'à l'étape 1
#     (generer_copies avec GRAINE_BRUIT), sans mise à jour, pour les deux méthodes ;
#   - tableau : score et décision sur les motifs propres, puis erreur à 10, 30, 50 % ;
#   - figure : perte L selon l'époque + courbes d'erreur au bruit des deux méthodes.

from apprentissage import entrainer
from etape1 import BIAIS_INITIAL, ETA, GRAINE_ORDRE, GRAINE_POIDS, MAX_EPOQUES
from fichiers import ecrire_csv
from lecture import lire_motif
from neurone import calculer_score, decider, initialiser_poids
from widrow_hoff import entrainer_wh, predire, stabilite

ETA_SECOURS = 0.001        # second essai si la perte devient non finie
EPOQUES_AFFICHEES = 5      # nombre de premières époques dont on affiche la perte


def main():
    motifs = [lire_motif("data/zero.txt"), lire_motif("data/un.txt")]

    # Contrôle de stabilité : eta * ||[x, 1]||² doit rester < 2
    for pixels, classe in motifs:
        print(f"chiffre {classe} : eta * ||[x,1]||² = {stabilite(pixels, ETA):.3f}")

    # Mêmes poids initiaux que l'étape 1 (même graine), pas ses poids appris
    poids_initiaux = initialiser_poids(48, GRAINE_POIDS)

    # Règle du perceptron (comme à l'étape 1)
    w_p, b_p, erreurs = entrainer(poids_initiaux, BIAIS_INITIAL, motifs, ETA, MAX_EPOQUES,
                                  GRAINE_ORDRE)
    print(f"\nPerceptron : {len(erreurs) - 1} époque(s), erreurs finales = {erreurs[-1]}")

    # Widrow–Hoff
    eta = ETA
    w_wh, b_wh, pertes, statut = entrainer_wh(poids_initiaux, BIAIS_INITIAL, motifs, eta,
                                              MAX_EPOQUES, GRAINE_ORDRE)
    if statut == "perte non finie":
        print(f"ATTENTION : perte non finie avec eta = {eta}. Nouvel essai avec eta = {ETA_SECOURS}.")
        eta = ETA_SECOURS
        w_wh, b_wh, pertes, statut = entrainer_wh(poids_initiaux, BIAIS_INITIAL, motifs, eta,
                                                  MAX_EPOQUES, GRAINE_ORDRE)
    print(f"Widrow–Hoff (eta = {eta}) : {len(pertes) - 1} époque(s), "
          f"perte finale L = {pertes[-1]:.2e} ({statut})")
    for epoque in range(min(EPOQUES_AFFICHEES + 1, len(pertes))):
        print(f"  époque {epoque} : L = {pertes[epoque]:.6f}")

    # Motifs propres : score, prédiction (seuil 0 pour le perceptron, 0,5 pour Widrow–Hoff)
    print("\n  méthode     | chiffre (target) | score    | prédiction")
    for pixels, target in motifs:
        score = calculer_score(w_p, b_p, pixels)
        print(f"  perceptron  | {target:16d} | {score:+.4f} | {decider(score)}")
    for pixels, target in motifs:
        score = calculer_score(w_wh, b_wh, pixels)
        print(f"  widrow-hoff | {target:16d} | {score:+.4f} | {predire(score)}")

    print()
    lignes = []
    for epoque in range(len(pertes)):
        lignes.append([epoque, pertes[epoque]])
    ecrire_csv("resultats/etape2_perte.csv", ["epoque", "perte"], lignes)


if __name__ == "__main__":
    main()
