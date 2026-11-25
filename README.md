# Colorisation guidée par clics (U-Net, DIV2K)

L'utilisateur clique quelques points sur une image en niveaux de gris et donne leur couleur. Le réseau colorise le reste.

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
- clic gauche : ajouter, clic droit : annuler
- clics sauvegardés dans clicks/<image>.json (supprimer le fichier pour refaire)
- chaque clic = carré 7 × 7 de la couleur moyenne autour du point
```

Lab converti à la volée (pas précalculé) : même résultat, ~1,5 Go de PNG au
lieu de ~7 Go de Lab, une seule source de données, même code à l'entraînement
et à l'inférence.

## Pourquoi la couleur reste près du point cliqué

- À l'entraînement, le modèle voit beaucoup d'indices (6 à 655 pixels par crop
  256 × 256 -- car on choisi le nombre de hints selon une densité suivant une loi
  log-uniforme) : la vraie couleur est toujours à ~5-10 px. Il a donc appris à
  recopier l'indice le plus proche sur quelques pixels, pas à propager loin.
- Avec 10 clics sur 678 × 1020, les clics sont à ~260 px les uns des autres :
  c'est un cas jamais vu.
- Un pixel de sortie ne « voit » qu'environ 100 px autour de lui (champ
  réceptif) : loin d'un clic, il ne voit aucune couleur et prédit du gris.
- Les clics sont des carrés 7 × 7, alors qu'à l'entraînement ce sont des
  pixels isolés.

Pour y remédier : simuler de vrais clics à l'entraînement (peu de patchs de
couleur moyenne) et agrandir le champ réceptif (cf. TODO, étapes 2 et 3).

## TODO (dans l'ordre)

1. [ ] **Métriques** PSNR / SSIM / LPIPS selon le nombre de clics : pour
   comparer chaque changement qui suit
2. [ ] **Simuler des clics à l'entraînement** au lieu d'une densité de pixels :
   nombre de clics géométrique (p=1/8, 0 inclus), patchs de 1×1 à 9×9 de couleur
   moyenne (Zhang 2017). Aujourd'hui le modèle voit 6 à 655 pixels isolés par
   crop, alors que 10 clics sur 678×1020 ≈ 1 patch 7×7 par crop : jamais vu, d'où
   la couleur qui reste autour du point
3. [ ] **Agrandir le champ réceptif** (~100 px aujourd'hui) : 4ᵉ niveau /
   maxpool en plus (en cours, donne de meilleurs résultats), sinon convs dilatées
   ou branche globale dans le bottleneck\
**Tout ce qui est en dessous est optionnel je pense :**
4. [ ] *(seulement si besoin)* Remplacer BatchNorm par GroupNorm : utile
   uniquement si le nombre de clics au test sort de la plage vue à
   l'entraînement (après l'étape 2 ce ne devrait plus être le cas)
5. [ ] *(optionnel)* Augmentation d'échelle : redimensionner aléatoirement
   (côté court entre 256 et la taille d'origine) avant le crop 256, pour être
   robuste à la taille des objets. Pas d'entraînement sur images seulement
   réduites : on infère en pleine résolution, les objets n'auraient pas la
   même taille
6. [ ] L1 pondérée par la saturation de la vérité terrain : la L1 seule tend
   vers la médiane ≈ gris quand le modèle hésite
7. [ ] Encodeur pré-entraîné ImageNet (resnet34, conv1 à 4 canaux, pad multiple
   de 32) : le U-Net from scratch sur 640 images ne sait pas ce qu'est un objet
   (couleurs à cheval sur deux objets)
8. [ ] Si encore terne : classification sur ab quantifié + rééquilibrage +
   annealed mean (Zhang 2016)
9. [ ] En dernier recours : perte adversariale type ChromaGAN (cf. `refs/`)
10. [ ] Régler les hyperparamètres en validation
11. [ ] Rangement : `normalize` / `denormalize` dans `MyDataset`, re-ranger les fichiers

## Historique des poids

| Poids | Commit | Indices à l'entraînement | Meilleure epoch | Val loss |
|---|---|---|---|---|
| `1_model20261117-1613.pt` | `50b473b` | masque 1 %, `n_aug=0` | 12/15 | 0.0524 |
| `model20261117-2301.pt` | `1697c79` | densité log-uniforme 0,01 %-1 %, `n_aug=1` | 9/12 | 0.0889 |
| `model20261125-1607.pt` | `e6dc0ba` | clics simulés : nombre ~ géométrique(p=1/4, 0 inclus), carrés 7×7 de couleur moyenne, `n_aug=1` | 26/29 | 0.0811 |

Communs : AdamW lr 3e-4, patience 3, L1, crop 256, split seed 42.
Batch 32 pour `1613` et `2301`, batch 16 pour `1607`.
Les val loss ne sont pas comparables entre elles (indices simulés différents).

- `1607` : **UNet à 4 niveaux (4 MaxPool)** au lieu de 3, soit un champ
  réceptif plus grand (~200 px au lieu de ~100) ; 124 Mo au lieu de 31 Mo.
  En test, les images doivent être paddées à un multiple de 16 (2⁴) au lieu
  de 8 : `PadToMultiple(16)`. Ces poids ne se chargent pas dans l'ancien UNet
  à 3 niveaux, et inversement.
  Valid à la meilleure epoch (≈ 26-29) : PSNR ≈ 21,3 dB, SSIM ≈ 0,92,
  LPIPS ≈ 0,18 (métriques en RGB, sur les crops 256 de validation).

- `1613` paraît meilleur car testé dans ses conditions d'entraînement (1 % =
  ~6900 points sur 678×1020, irréaliste pour un utilisateur) ; à 0,01 % il
  sort une image grise.
- `2301` est plus honnête mais terne à faible densité, et étale trop les
  couleurs à 1 % (bord de sa plage, aggravé par la BatchNorm).

## Dataset

```
./src/scripts/download_data.sh
```

- [DIV2K train LR x2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)
- [DIV2K valid LR x2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)

## Ancien TODO

- [x] transformer les images en couleurs en niveau de gris (gratuit) --> pour données d'entrainement --> utiliser div2k
- [x] pour le scribbles: laisser 1% de couleur dans les images grises pour prédire le reste. (couleurs réelles) (pour entrainement)
- [ ] à l'inférence c'est l'utilisateur qui donne les couleurs
- [ ] metrics psnr, signature noise metric

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

## UPDATE UpConv

Pad de `x1` après l'upsampling pour combler l'écart de taille avec `x2` (skip connection).

```python
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], 1) # NOTE : /!\ on colle la conv du down par la droite, vaut-il mieux que ce soit par la gauche ??
        return self.conv(x)
```




Le principe

1. Découper l'image en superpixels avec skimage.segmentation.slic. On obtient quelques centaines de petites zones de couleur homogène.
2. Construire le RAG avec rag_mean_color : chaque superpixel devient un nœud, deux superpixels voisins sont reliés par une arête, et le poids de l'arête est la différence entre leurs couleurs moyennes.
3. Fusionner les régions semblables, avec cut_threshold (on coupe les arêtes au-dessus d'un seuil) ou merge_hierarchical. On obtient quelques grandes régions de couleur homogène.
4. Placer un clic par région, avec la couleur (a, b) moyenne du patch autour du point.

