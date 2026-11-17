- [x] transformer les images en couleurs en niveau de gris (gratuit) --> pour données d'entrainement --> utiliser div2k
- [x] pour le scribbles: laisser 1% de couleur dans les images grises pour prédire le reste. (couleurs réelles) (pour entrainement)
- [ ] à l'inférence c'est l'utilisateur qui donne les couleurs
- [ ] metrics psnr, signature noise metric

## TODO
- [x] show_image
- [x] refaire proprement les bons noms (train / valid) puis test
- [ ] re ranger proprement les fichiers
- [ ] entrainement avec bcp plus d'EPOCHS (100 ?) car sinon 800 * 0.8 = 640 images en training (640 / 32 batchs = 20 pas par epochs) si on fait que 5 epochs c'est que 100 pas
- [ ] Dans base.transforms ==> mettre transforms.RandomCrop(256) ==> patch de 256 de l'image 
      (chaque pas voit des patchs différents (donc bcp + de données) + image pas écrasée + détails conservés)
- [x] MSELoss donne des couleurs saturées, utiliser nn.L1Loss ou nn.SmoothL1Loss
- [x] Ajout batchNorm2d pour accélérer la convergence du UNet
- [ ] densité de hints variable à l'entraînement (sinon modèle voit que du 1% et sera perdu)
- [ ] créer une fonction normalize et déplacer la fonction denormalize dans MyDataset
- [x] Mettre des num_workers dans les DataLoaders
- [x] factoriser show_image avec MyDataset
- [x] Dans MyDataset, un paramètre de mode :
    - [x] train: RandomCrop + augmentations + Normalize + hints aléatoires (densité variable)
    - [x] valid : CenterCrop 256 (ou image entière), pas d'augmentation, Normalize + hints aléatoires
    - [x] test : image entière + pad pour être un multiple de 8, pas d'augmentation, Normalize + hints tirés avec une graine fixe (pour PSNR / SSIM reproductibles)
    (multiple de 8 car on fait 3 maxpool -- on a divisé l'image par 2, 3 fois donc 2^3=8)

# autres approches
- scribbles 
- colorization à partir de ref (volorimétrie d'un animal sur une voiture par ex) ---- U-net
- reseau de neuronnes profonds

## Pipeline

```
DIV2K (800 images couleur en train, 100 en validation)
- passage en Lab : L = entrée, (a, b) = cible

- carrés de 256 × 256 tirés au hasard, retournements (augmentation)

- scribbles simulés: quelques pixels tirés au hasard gardent leur vraie
  couleur (a, b) /!\ un masque vaut 1 sur ces pixels, 0 ailleurs

- U-Net : entrée 4 canaux = L + (a, b) des scribbles + masque PUIS sortie (a, b)

- perte entre (a, b) prédits et vrais, sur toute l'image

- on recolle L d'origine + (a, b) prédits → conversion en RVB

- évaluation : PSNR, SSIM, LPIPS selon le nombre de scribbles, et regarder
  les images

A l'inférence : c'est l'utilisateur qui place les points.
```

## choix de convertir en lab à la volée

- Même résultat : le recadrage et le retournement donnent la même chose avant
  ou après la conversion en Lab. Précalculer ne fait donc rien gagner en
  qualité.
- Coût négligeable : rgb2lab sur un carré 256×256 prend quelques millisecondes.
  Avec plusieurs workers dans le DataLoader, c'est caché derrière
  l'entraînement sur GPU.
- Moins d'espace disque : le Lab en float32 prend ~7 Go, contre ~1,5 Go pour
  les PNG d'origine.
- Une seule source de données : pas de deuxième copie du dataset à régénérer et
  à garder synchronisée si on change quelque chose.
- Plus souple : on peut ajouter des augmentations de couleur (luminosité,
  saturation…), qui se font en RVB, ou changer la normalisation, sans tout
  recalculer.
- Même code pour l'entraînement et l'inférence : à l'inférence, l'utilisateur
  fournit une image RVB, il faut donc de toute façon convertir à la volée.
- Pipeline plus simple : une étape de préparation en moins, plus facile à
  expliquer dans le README.

## Dataset

[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)
[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

To download the dataset, we set up a install and preparation scripts

```
./src/scripts/download_data.sh && python ./src/scripts/to_grayscale.py
```
