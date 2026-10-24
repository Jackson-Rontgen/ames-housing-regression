# Prédire le prix des maisons (Ames Housing)

Projet de régression fait pendant mon M1 de mathématiques appliquées et statistique. Je prédis le prix de vente d'une maison à partir de ses caractéristiques (surface, qualité, année de construction, quartier, etc.).

L'énoncé demandait une régression linéaire simple et multiple, évaluée avec le R², le MSE, le RMSE, le MAPE et l'analyse des erreurs. Je suis parti de là, puis j'ai ajouté ElasticNet parce que ma régression multiple faisait du sur-apprentissage.

## Les données

Les données viennent de la compétition Kaggle « House Prices : Advanced Regression Techniques » (maisons vendues à Ames, dans l'Iowa). Le fichier `train.csv` contient 1 460 maisons et 81 colonnes : l'identifiant, 79 variables et le prix de vente (`SalePrice`, ma variable cible). Mon notebook lit aussi le fichier `data_description.txt`, qui explique chaque colonne.

Ces deux fichiers ne sont pas dans ce dépôt : il faut les télécharger sur Kaggle et les mettre à côté du notebook.

## Comment j'ai nettoyé la base

Le nettoyage est la partie qui m'a demandé le plus de réflexion. Pour chaque colonne, je me suis demandé : est-ce que la valeur manque vraiment, ou est-ce que « NA » veut dire que la chose n'existe pas ? Chaque choix est écrit ci-dessous.

### Exploration de départ
- 1 460 lignes, 81 colonnes (35 entiers, 3 décimaux, 43 textes).
- Aucune ligne en double.
- Le prix moyen est de 180 921 $ et le prix médian de 163 000 $. La moyenne est plus haute parce que quelques maisons très chères tirent le prix vers le haut. Je l'ai vérifié avec l'histogramme du prix et celui de son logarithme.
- Les variables les plus corrélées au prix sont `OverallQual` (la qualité générale, 0,79) et `GrLivArea` (la surface habitable, 0,71).

### Valeurs aberrantes
J'ai regardé les nuages de points entre le prix et les variables les plus corrélées. Deux maisons ne ressemblaient à aucune autre : plus de 4 000 pieds carrés de surface habitable, mais vendues moins de 200 000 $ (184 750 $ et 160 000 $). Avec une surface pareille, elles auraient dû coûter bien plus cher. **Je les ai supprimées**, pour qu'elles ne tirent pas la droite de régression vers le bas.

J'ai aussi regardé les maisons avec un garage de plus de 3 places vendues sous 200 000 $. Je les ai **gardées** : ce sont des maisons modestes avec un grand garage, pas des erreurs.

### Valeurs manquantes
Pour chaque colonne qui avait des valeurs manquantes, j'ai appliqué une des règles suivantes.

| Colonnes | Ce que j'ai fait | Pourquoi |
|---|---|---|
| `Electrical` (1 ligne) | Suppression de la ligne | Une seule maison sur 1 460, ça ne change rien. |
| `MasVnrArea` | Remplacée par 0 | Une valeur manquante veut dire « pas de placage en maçonnerie », donc une surface de 0 est la vraie valeur. |
| `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | Remplacées par « None » | Pas de sous-sol. |
| `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | Remplacées par « None » | Pas de garage. |
| `GarageYrBlt` | Remplacée par l'année de construction de la maison (`YearBuilt`) | Un garage « construit en l'an 0 » n'a pas de sens, et je ne voulais pas supprimer les maisons sans garage. |
| `FireplaceQu`, `MasVnrType`, `MiscFeature`, `Fence`, `Alley`, `PoolQC` | Remplacées par « None » | Pas de cheminée, pas de clôture, pas de piscine, etc. Certaines de ces colonnes ont plus de 90 % de valeurs manquantes : supprimer les lignes m'aurait laissé presque aucune maison. |
| `LotFrontage` (longueur de la façade sur la rue) | Médiane du quartier (`Neighborhood`) | Cette fois la valeur manque vraiment. Les maisons d'un même quartier ont des façades de taille proche. |

Je n'ai **jamais** rempli avec 0 quand 0 n'avait pas de sens. Mon erreur au début a été de le faire pour `LotFrontage` et `GarageYrBlt` : une façade de 0 pied ou un garage construit en l'an 0 faussaient la corrélation avec le prix.

### Types et variables
- `MSSubClass` est un code (type de logement), pas une quantité : je l'ai transformée en texte.
- Les quatre colonnes de salles de bain (`FullBath`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath`) sont regroupées en une seule : `TotalBath`, où une demi-salle de bain compte pour 0,5.
- J'ai supprimé `GarageCars` : elle est presque la même chose que `GarageArea` (corrélation d'environ 0,9). Je garde une seule des deux, la plus précise, pour ne pas donner deux fois la même information au modèle.
- J'ai retiré `Id`, qui ne décrit pas la maison.
- Les colonnes de texte sont transformées en colonnes 0/1 avec `get_dummies(drop_first=True)`. Une catégorie est retirée à chaque fois et sert de référence.

**Résultat : 1 457 maisons et 267 colonnes explicatives** (1 460 maisons au départ, moins 2 aberrantes, moins 1 sans `Electrical`).

## Ce que j'ai fait ensuite

1. **Découpage.** 80 % des maisons pour entraîner (1 165) et 20 % pour tester (292), avec `random_state=42`. Je ne regarde le test qu'à la fin.
2. **Régression simple** avec la variable la plus corrélée : `OverallQual`.
3. **Régression à deux variables** : `OverallQual` et `GrLivArea`.
4. **Régression linéaire multiple** avec les 267 colonnes, après `StandardScaler`. C'est ma référence.
5. **ElasticNet** (`alpha = 1`, `l1_ratio = 0,5`) dans un `Pipeline` avec `StandardScaler`.
6. **ElasticNet + GridSearchCV** : je teste 36 réglages d'`alpha` et de `l1_ratio` avec une validation croisée à 5 plis, et je garde celui qui a la plus petite erreur (RMSE). Le test n'est jamais utilisé pour choisir.

## Résultats

### Modèles simples (jeu de test)

| Modèle | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| 1 variable (`OverallQual`) | 44 017 $ | 32 246 $ | 20,1 % | 0,69 |
| 2 variables (`OverallQual`, `GrLivArea`) | 38 456 $ | 28 762 $ | 18,6 % | 0,76 |

### Modèles avec toutes les variables

| Modèle | RMSE train | RMSE test | R² train | R² test |
|---|---|---|---|---|
| Régression linéaire | 18 822 $ | 27 210 $ | 0,94 | 0,88 |
| ElasticNet (alpha = 1, l1_ratio = 0,5) | 23 324 $ | 24 689 $ | 0,91 | 0,90 |
| ElasticNet + GridSearchCV | 20 746 $ | 23 678 $ | 0,93 | 0,91 |

Meilleurs réglages trouvés : `alpha = 3` et `l1_ratio = 0,95`.

Mon meilleur modèle se trompe de 23 678 $ en moyenne (RMSE), soit environ 15 % du prix médian (163 000 $).

## Mon interprétation

### Sur-apprentissage et ElasticNet
La régression linéaire à 267 colonnes fait beaucoup mieux sur l'entraînement (18 822 $ d'erreur) que sur le test (27 210 $). C'est du **sur-apprentissage** : avec 267 colonnes pour 1 165 maisons, le modèle a trop de liberté et il apprend le train par cœur.

ElasticNet corrige ça en pénalisant les gros coefficients. L'erreur sur le train monte un peu (20 746 $) mais celle sur le test baisse (23 678 $, soit environ 13 % de moins que la régression linéaire). L'écart entre train et test devient bien plus petit. Le R² de test passe de 0,88 à 0,91.

### Les modèles simples
- Avec une seule variable, un point de qualité de plus est associé à environ 45 748 $ de plus. Ce chiffre mélange l'effet de la qualité et celui des variables qui vont avec (une maison de meilleure qualité est aussi souvent plus grande).
- Avec deux variables, un point de qualité de plus vaut environ 31 515 $ de plus **à surface égale**, et un pied carré de plus vaut environ 68 $ de plus **à qualité égale**. Les coefficients changent parce qu'on contrôle l'autre variable.
- Je n'interprète pas l'ordonnée à l'origine (environ −98 499 $) : `OverallQual = 0` n'existe pas dans les données, elle sert seulement à placer la droite.

### Résidus de la régression simple
Les points forment des **colonnes verticales** : `OverallQual` ne prend que des valeurs entières de 1 à 10, donc le modèle ne prédit qu'une dizaine de prix différents. Dans chaque colonne, l'erreur va à peu près de −90 000 à +100 000 $ : à qualité égale, les prix varient beaucoup, donc la qualité seule ne suffit pas. Deux points sont même prédits à un **prix négatif**, pour les qualités les plus basses : c'est une limite d'une droite avec une ordonnée à l'origine très négative.

### Prédit contre vrai (ElasticNet réglé)
Les points sont serrés autour de la diagonale pour les maisons de moins de 300 000 $ : le modèle suit bien. Au-delà, les points s'écartent. La maison la plus chère du test (environ 625 000 $) est prédite à environ 485 000 $ : le modèle **sous-estime** les maisons très chères, parce qu'il en a très peu vu.

### Résidus (ElasticNet réglé)
Les résidus sont centrés sur 0 et ne dessinent pas de courbe, donc le modèle n'a pas d'erreur systématique. Par contre, ils forment un **entonnoir** : ils restent autour de ±25 000 $ pour les petits prix et montent jusqu'à ±70 000 $ pour les prix prédits au-dessus de 300 000 $. L'erreur grandit avec le prix : le modèle se trompe plutôt en pourcentage qu'en dollars. C'est une raison d'essayer le logarithme du prix. Le point isolé à +140 000 $ est la maison à 625 000 $.

### Les 15 variables qui pèsent le plus
Les variables sont standardisées : un coefficient se lit comme « le prix change de tant de dollars quand la variable augmente d'un écart-type, les autres restant fixes ».
- **Les surfaces dominent** : `GrLivArea` (environ +11 700 $), `TotalBsmtSF`, `1stFlrSF`, `BsmtFinSF1`, `2ndFlrSF`. Ces variables sont liées entre elles (la surface habitable est à peu près la somme des surfaces du rez-de-chaussée et de l'étage). Le modèle partage l'effet entre elles : je ne peux pas dire que l'une compte plus que l'autre, seulement que « la surface » compte beaucoup.
- **La qualité générale** (`OverallQual`, environ +9 300 $) est la deuxième.
- **Le quartier** compte : `Neighborhood_StoneBr` et `Neighborhood_NridgHt` font monter le prix.
- Trois colonnes ont un coefficient négatif : `KitchenQual_Gd`, `KitchenQual_TA` et `BsmtQual_Gd`. À cause de `drop_first=True`, elles sont comparées à la catégorie retirée, « Ex » (excellent). Ça veut dire : une cuisine « bonne » ou « typique » vaut moins qu'une cuisine « excellente », à autres caractéristiques égales. Ça ne veut pas dire qu'une meilleure cuisine fait baisser le prix.
- Ce sont des associations, pas des causes.

### Comparaison des modèles
L'erreur de test baisse à chaque étape : 44 017 $ (1 variable), 38 456 $ (2 variables), 27 210 $ (régression linéaire complète), 24 689 $ (ElasticNet), 23 678 $ (ElasticNet réglé).

## Limites
- **Un seul découpage train/test** : le résultat pourrait changer un peu avec un autre `random_state`.
- **Fuite de données** : j'ai rempli `LotFrontage` avec la médiane du quartier avant de séparer train et test, donc le test a influencé un peu mon train. L'effet est sûrement petit, mais il faudrait le faire après le découpage, dans un `Pipeline`.
- **`alpha = 3` est le bord de ma grille** (de 0,01 à 3) : un `alpha` encore plus grand pourrait être meilleur. Je ne l'ai pas testé.
- **Avertissements de convergence** : pour les petits `alpha`, ElasticNet affiche « Objective did not converge » même avec 10 000 itérations.
- **Erreurs qui grandissent avec le prix** (entonnoir des résidus) : je n'ai pas essayé de passer le prix au logarithme.
- **Maisons très chères sous-estimées** : peu d'exemples au-dessus de 400 000 $.
- **MAE et MAPE** ne sont calculés que pour les deux modèles simples.
- **Le MAPE** pèse plus lourd les erreurs sur les maisons peu chères. **Le RMSE** pèse plus lourd les grosses erreurs, donc les maisons chères.

## Pistes pour la suite
Passer le prix au logarithme et comparer les erreurs en dollars, essayer des `alpha` plus grands, calculer MAE et MAPE pour tous les modèles, créer une variable de surface totale, et tester d'autres modèles (forêts aléatoires, par exemple).

## Lancer le projet
Python 3 avec `pandas`, `numpy`, `matplotlib`, `seaborn` et `scikit-learn`. Mettre `train.csv` et `data_description.txt` dans le même dossier que le notebook, puis faire `Restart & Run All`.
