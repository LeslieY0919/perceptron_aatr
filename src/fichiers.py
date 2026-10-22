# Petites fonctions pour enregistrer les résultats (CSV, JSON) dans resultats/.

import csv
import json
import os


def ecrire_csv(chemin, entete, lignes):
    """Écrit un fichier CSV : une ligne d'en-tête, puis une ligne par élément de `lignes`."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(entete)
        for ligne in lignes:
            ecrivain.writerow(ligne)
    print(f"  écrit : {chemin}")


def ecrire_json(chemin, donnees):
    """Enregistre un dictionnaire (réglages, poids...) au format JSON."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier)
    print(f"  écrit : {chemin}")


def lire_json(chemin):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)
