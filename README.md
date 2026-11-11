- [x] transformer les images en couleurs en niveau de gris (gratuit) --> pour données d'entrainement --> utiliser div2k
- [ ] pour le scribbles: laisser 1% de couleur dans les images grises pour prédire le reste. (couleurs réelles) (pour entrainement)
- [ ] à l'inférence c'est l'utilisateur qui donne les couleurs
- [ ] metrics psnr, signature noise metric

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


## Dataset

[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)
[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

To download the dataset, we set up a install and preparation scripts

```
./src/scripts/download_data.sh && python ./src/scripts/to_grayscale.py
```
