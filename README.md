- [x] transformer les images en couleurs en niveau de gris (gratuit) --> pour données d'entrainement --> utiliser div2k
- [x] pour le scribbles: laisser 1% de couleur dans les images grises pour prédire le reste. (couleurs réelles) (pour entrainement)
- [ ] à l'inférence c'est l'utilisateur qui donne les couleurs
- [ ] metrics psnr, signature noise metric

## TODO
- [x] show_image
- [x] refaire proprement les bons noms (train / valid) puis test
- [ ] re ranger proprement les fichiers
- [x] entrainement avec bcp plus d'EPOCHS et early stopping (20) car sinon 800 * 0.8 = 640 images en training (640 / 32 batchs = 20 pas par epochs) si on fait que 5 epochs c'est que 100 pas
- [x] Dans base.transforms ==> mettre transforms.RandomCrop(256) ==> patch de 256 de l'image 
      (chaque pas voit des patchs différents (donc bcp + de données) + image pas écrasée + détails conservés)
- [x] MSELoss donne des couleurs saturées, utiliser nn.L1Loss ou nn.SmoothL1Loss
- [x] Ajout batchNorm2d pour accélérer la convergence du UNet
- [x] densité de hints variable à l'entraînement (sinon modèle voit que du 1% et sera perdu)
- [ ] créer une fonction normalize et déplacer la fonction denormalize dans MyDataset
- [x] Mettre des num_workers dans les DataLoaders
- [x] factoriser show_image avec MyDataset
- [x] Dans MyDataset, un paramètre de mode :
    - [x] train: RandomCrop + augmentations + Normalize + hints aléatoires (densité variable)
    - [x] valid : CenterCrop 256 (ou image entière), pas d'augmentation, Normalize + hints aléatoires
    - [x] test : image entière + pad pour être un multiple de 8, pas d'augmentation, Normalize + hints tirés avec une graine fixe (pour PSNR / SSIM reproductibles)
    (multiple de 8 car on fait 3 maxpool -- on a divisé l'image par 2, 3 fois donc 2^3=8)
- [ ] faire de la cross validation ?
- [ ] régler hyperparamètres en validation
- [ ] inférer en basse résolution (réduire ×2-4, prédire ab, remonter ab en bilinéaire, recoller le L d'origine) : le champ réceptif (~96 px) couvre alors plus de scène, teste l'hypothèse sans réentraîner
- [ ] simuler des clics au lieu d'une densité (nombre géométrique p=1/8, 0 inclus) : 10 clics sur 678×1020 ≈ 1 hint par crop 256², jamais vu à l'entraînement (min ≈ 6,5)
- [ ] hints en petits patchs 1×1 à 9×9 de couleur moyenne (Zhang 2017) : un pixel isolé est noyé dès la 1ʳᵉ conv, un clic utilisateur est une zone
- [ ] `show_image(..., n_clicks=10)` au lieu de `p` : un nombre de clics a le même sens quelle que soit la taille de l'image
- [ ] entraîner sur images réduites (côté court aléatoire 256-400 puis crop 256) : le crop voit presque toute la scène, cohérent avec l'inférence basse résolution
- [ ] agrandir le champ réceptif (4ᵉ niveau UNet, convs dilatées ou branche globale dans le bottleneck) : un clic ne peut pas colorer un train de ~800 px avec un champ réceptif de ~96 px
- [ ] encoder pré-entraîné ImageNet (resnet34 torchvision, conv1 passée à 4 canaux, pad multiple de 32) : le UNet from scratch sur 640 images n'apprend pas ce qu'est un objet, d'où des couleurs à cheval sur deux objets (chapeau bleu/beige)
- [ ] remplacer BatchNorm par GroupNorm : les stats figées en eval ne s'adaptent pas au nombre de hints de test
- [ ] L1 pondérée par la saturation de la vérité terrain : la L1 seule tend vers la médiane ≈ gris quand le modèle hésite, d'où les couleurs ternes
- [ ] si encore terne, classification sur ab quantifié + rééquilibrage + annealed mean (Zhang 2016) : force le modèle à choisir une couleur au lieu de moyenner
- [ ] en dernier recours, perte adversariale type ChromaGAN (cf. `refs/`) : couleurs plus réalistes mais entraînement plus lourd
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


## Historique des poids
- `model20261117-2301.pt` : modèle avec des résultats honnêtes (terne à faible densité, couleurs fausses à 1 % -- car moins de hints au training)
  - commit `1697c7948dcb1024e8060558815ad50fc4cfe96f`
  - lr 3e-4 (AdamW), batch 32, 20 epochs max, patience 3, loss L1
  - crop 256, `n_aug=1`, masque selon densité log-uniforme entre 0.01% et 1 %, split 80/20, seed 42
  - meilleure epoch 9/12, val loss 0.0889


- `model20261117-1613.pt` : Premier modèle avec des résultats convaincants
  - commit `50b473bfc05184ce04456ca13f3aada283cc8a1f`
  - lr 3e-4 (AdamW), batch 32, 20 epochs max, patience 3, loss L1
  - crop 256, `n_aug=0`, masque 1 %, split 80/20 seed 42
  - meilleure epoch 12/15, val loss 0.0524
  - seul le split est seedé : réentraîner donne des poids proches, pas identiques

## Dataset

[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)
[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

To download the dataset, we set up a install and preparation scripts

```
./src/scripts/download_data.sh && python ./src/scripts/to_grayscale.py
```

## Pourquoi l'ancien modèle semblait meilleur

- ancien modèle `model20261117-1613.pt` est trained et tested avec 1% de
  pixels-hints --> conditions identiques d'où les couleurs vives et cohérentes
    - Cependant 1 % d'une image 678x1020, c'est 6915 points --> pas réaliste
      pour un user à l'inférence --> donc modèle spécialisé sur un cas
      irréaliste
    - j'ai testé à l'inférence avec p=1e-4=0.01% (soit 1020x678 x1e-4 = 69
      hints placés), on a une image bien grise avec qlq pts de couleurs

- Le modèle actuel `model20261117-2301.pt` est trained avec une densité log
  uniforme entre 0.01% et 1%. 
    - Testé à 1%: il est au bord de sa plage d'entraînement: il étale trop les
      couleurs (bleu sur les montagnes) et la BatchNorm aggrave ça : en eval
      elle normalise avec des statistiques figées apprises sur des densités
      surtout faibles, donc elle ne s'adapte pas à 1 % d'où des valeurs a/b
      extrêmes
    - Testé à une densité réaliste (0.01%), il donne des couleurs plus ternes
      --> ce modèle est moins beau mais plus honnête
    - Attention: 0.01% est aussi un bord de la plage d'entraînement, le test
      juste est au milieu (0.1%, ≈ 690 hints)
- La val loss actuelle (0,0889) n'est pas comparable à l'ancienne (0,0524): la
  densité moyenne des hints n'est pas la même.

<!-- - Pistes : une plage de densités centrée sur la densité de test (ex. 0,1 %
à 3 %) et des indices en petits carrés plutôt qu'en pixels isolés, comme dans
Zhang et al. 2017. -->
