# perceptron_aatr

TD « IA pour les systèmes embarqués » : du perceptron à MNIST.
Les calculs sont écrits à la main, avec des boucles, sans NumPy.

## Avancement

Étape 1 — un neurone distingue 0 et 1 :

- [x] fichiers `data/zero.txt` et `data/un.txt`, lecture des 48 pixels
- [ ] score, décision, mise à jour des poids
- [ ] apprentissage par époques
- [ ] test au bruit
- [ ] CSV, figure, analyse

## Lancer

Depuis la racine du dépôt :

```bash
python3 src/lecture.py
python3 -m unittest discover -s tests -v
```
