# Système agentique de préparation de données pour le ML

## 1. Objectif

Automatiser la préparation de données tabulaires pour le machine learning. Le système prend en entrée un `.csv` de données brutes et produit des jeux `train` / `val` / `test` prêts pour une phase ultérieure de normalisation et d'entraînement.

Le système exécute **trois phases en série** :

1. **Data cleaning** — agent qui nettoie les données
2. **Feature engineering** — agent qui construit/transforme les features
3. **Data splitting** — découpage déterministe en train/val/test

## 2. Principe fondamental : zéro fuite de données (data leakage)

> **Règle non négociable.** Aucune transformation susceptible de provoquer une fuite vers les données de validation ou de test n'est autorisée dans les phases 1 et 2. Le split (phase 3) intervient **après** cleaning et feature engineering ; toute transformation appliquée avant le split doit donc être **stateless / row-local**.

### 2.1 Définition

- **Transformation autorisée (stateless / row-local).** Le résultat pour une ligne ne dépend **que** des valeurs de cette même ligne. Aucune statistique calculée sur l'ensemble (ou un sous-ensemble) du dataset n'est utilisée.
- **Transformation interdite avant le split (stateful / fitted).** La transformation dépend d'une statistique agrégée du dataset (moyenne, médiane, quantile, fréquence, corrélation, composantes, vocabulaire appris, etc.). Ces transformations sont **repoussées à la phase post-split** (non couverte par ce système), où elles seront *fittées sur `train` uniquement* puis appliquées à `val`/`test`.

### 2.2 Liste explicite — AUTORISÉ (stateless)

- Cast et correction de types (`str` → `datetime`, `object` → `category`, etc.)
- Parsing de dates et extraction de composantes locales (jour de semaine, mois, heure, weekend oui/non) à partir d'une colonne date de la **même ligne**
- Déduplication exacte de lignes
- Suppression de colonnes par règle structurelle indépendante des données (colonne 100 % constante, colonne ID, colonne entièrement vide)
- Nettoyage de chaînes par ligne (trim, lowercasing, regex, normalisation Unicode)
- Features dérivées purement locales : ratio/différence/produit de deux colonnes de la même ligne, comptage de tokens, flags conditionnels (`x > seuil_constant`)
- Remplacement par une **constante littérale fixée d'avance** (ex. `NaN → 0` *si et seulement si* la constante ne dépend d'aucune statistique du dataset, et que ce choix est documenté)

### 2.3 Liste explicite — INTERDIT avant le split (stateful / fitted)

- Imputation par moyenne / médiane / mode **calculés sur les données**
- Normalisation / standardisation / min-max / robust scaling
- Target encoding, frequency encoding, count encoding
- One-hot/label encoding dont le **vocabulaire est appris** sur l'ensemble du dataset
- Suppression d'outliers par IQR, z-score ou tout seuil dérivé d'une statistique globale
- Binning à quantiles, discrétisation basée sur la distribution
- Sélection de features par corrélation/importance vis-à-vis de la target
- PCA, embeddings appris, toute réduction de dimension fittée
- Resampling type SMOTE / undersampling/oversampling **basé sur la distribution**

> En cas de doute sur une transformation, l'agent doit la traiter comme **interdite** et la documenter comme « à réaliser post-split ».

## 3. Entrées

Le système reçoit une configuration explicite (ne **rien** inférer silencieusement) :

| Champ | Type | Description |
|---|---|---|
| `input_csv` | path | Chemin du `.csv` de données brutes |
| `target_column` | str | Nom de la colonne cible — **jamais** traitée comme une feature, jamais transformée, jamais utilisée comme source d'une feature |
| `task_type` | enum | `classification` \| `regression` — conditionne la stratification du split et les vérifications |
| `output_dir` | path | Répertoire de sortie des artefacts |

La `target_column` est isolée dès le chargement et protégée contre toute transformation des phases 1 et 2.

## 4. État partagé entre phases

Un objet d'état unique circule à travers le pipeline (modèle Pydantic v2). Il contient au minimum :

- Référence vers le dataframe courant (ou son chemin)
- `target_column`, `task_type`
- Schéma courant (colonnes + dtypes)
- **Historique des transformations** (voir §7)
- Métadonnées de provenance (hash du CSV source, seed, versions)

Chaque phase reçoit cet état, le valide en entrée, le mute de façon contrôlée, et le re-valide en sortie.

## 5. Contrat agent ↔ exécution déterministe

> **L'agent (LLM) décide ; le code déterministe exécute.** Aucun agent ne manipule directement le dataframe.

Pour chaque opération :

1. L'agent **décide** quelle transformation appliquer et avec quels paramètres.
2. L'agent **émet un script Python autonome et déterministe** (ou paramètre une fonction de transformation pré-existante).
3. **Gate obligatoire (§6).** Le script est soumis au **test d'invariance par ligne**. Tant qu'il n'a pas réussi ce test, il n'est **ni appliqué ni enregistré**. L'avis de l'agent sur le caractère « stateless » d'une transformation n'a aucune valeur de garantie : c'est ce test déterministe qui tranche.
4. Le système **exécute** le script validé de façon isolée, l'**enregistre** (versionné), et l'ajoute à l'historique.
5. Le résultat (script + params) est **rejouable** à l'identique sur de nouvelles données.

Cela garantit que la transformation des données reste reproductible — et stateless — même si le raisonnement de l'agent ne l'est pas. La garantie anti-leakage ne repose **jamais** sur le jugement du LLM, toujours sur le gate du §6.

## 6. Test d'invariance par ligne (gate obligatoire)

> **Chaque transformation des phases 1 et 2 DOIT impérativement réussir ce test avant d'être appliquée et enregistrée. Aucune exception.** C'est le mécanisme qui transfère la garantie « stateless » du jugement (non déterministe) du LLM vers une vérification (déterministe) par le code.

### 6.1 Principe

Une transformation `f` est row-local si et seulement si la valeur produite pour une ligne donnée **ne dépend pas des autres lignes présentes** dans le dataframe. Cette propriété est testable automatiquement : on applique `f` dans plusieurs contextes (sous-ensembles différents, éventuellement ordre mélangé), puis on vérifie que pour chaque ligne identifiable, **la sortie est strictement identique** quel que soit le contexte. Toute dépendance à une statistique agrégée (médiane, fréquence, quantile, etc.) brise cette identité et fait échouer le test.

> **Coût.** Le test est une vérification de *propriété*, pas l'application réelle de la transformation. Comme une transformation stateless l'est aussi sur n'importe quel sous-ensemble, les vérifications tournent sur un **échantillon de taille bornée** (ex. 2 000–5 000 lignes), **pas sur les N lignes**. Le seul passage en `O(N)` est le calcul de la sortie de référence `R = f(df)` — qui est de toute façon le travail réel à appliquer, pas un surcoût du gate. Le coût propre du test est donc une **constante indépendante de N**.

### 6.2 Procédure

Pour un script candidat `f`, on travaille sur un **échantillon borné** `E` tiré de `df` (taille fixe, ex. ≤ 5 000 lignes). Indexer chaque ligne de `E` par une clé stable (index d'origine ou hash du contenu de la ligne d'entrée).

**Noyau obligatoire — multi-sous-échantillons.**

1. **Référence.** Calculer `R = f(E)`.
2. **Sous-échantillons.** Tirer **au moins deux** sous-ensembles `S₁, S₂ ⊂ E` à fractions et seeds différents (ex. 40 % et 70 %), calculer `f(S₁)` et `f(S₂)`. Pour chaque ligne survivante, la sortie doit être **identique** à celle de `R`.

Deux tirages distincts perturbent à la fois le **multiset de valeurs** (attrape les statistiques agrégées : médiane, moyenne, mode, scaling, quantiles, IQR/z-score, target/frequency/count encoding, sélection par corrélation, PCA) et le **voisinage** des lignes survivantes (attrape partiellement les dépendances d'ordre). Cela couvre la quasi-totalité de la liste §2.3.

**Optionnel conditionnel — permutation.**

3. **Permutation.** Mélanger l'ordre des lignes de `E`, calculer `f(E_permuté)`, réaligner par clé, exiger l'identité avec `R`. Ce test cible spécifiquement les transformations **dépendantes de l'ordre ou du voisinage** (`shift`, `diff`, rolling window, `cumsum`, ffill, lag features) — un canal orthogonal au multiset, qu'une permutation garantit de façon fiable. **À activer uniquement si la phase FE est susceptible de produire des features ordonnées/temporelles** ; inutile sinon.

La transformation **réussit** si et seulement si toutes les comparaisons actives sont identiques (égalité exacte ; pour les flottants, tolérance nulle ou strictement bornée et justifiée).

### 6.3 Cas particuliers à gérer

- **Suppression de lignes (déduplication, filtrage).** Le test porte sur les lignes **survivantes** : leurs valeurs transformées doivent être invariantes au contexte. La décision de *garder ou non* une ligne doit elle-même être row-local (un filtre `x > seuil_constant` passe ; un filtre « top 10 % » échoue).
- **Suppression de colonnes structurelle.** Une règle comme « drop si la colonne est 100 % constante » dépend par nature de l'ensemble de la colonne. Elle est admise **uniquement** si la décision de drop est figée comme **paramètre déterministe de la recette** (liste explicite des colonnes à supprimer, déterminée une fois, puis rejouée à l'identique) — et non recalculée à chaque exécution. Le test d'invariance s'applique alors à la transformation paramétrée, pas à la détection.
- **Clé d'alignement.** Si `f` modifie les colonnes servant à identifier les lignes, indexer sur l'index positionnel d'origine capturé avant transformation.

### 6.4 Conséquence en cas d'échec

Si le gate échoue, le script est **rejeté**, non appliqué, non enregistré. Le système renvoie à l'agent un diagnostic (quelle vérification a échoué, exemple de ligne dont la sortie a varié) afin qu'il propose une alternative row-local ou marque l'opération comme « à réaliser post-split ». Le pipeline ne progresse pas tant qu'aucune transformation valide n'est produite ou que l'étape n'est pas explicitement abandonnée.

## 7. Historique des transformations (rejouable)

L'historique n'est **pas un simple log en prose** : c'est une **recette exécutable et déterministe**, sérialisée, ré-applicable telle quelle à de nouvelles données (inférence / production).

Pour chaque étape, enregistrer :

- `phase` (`cleaning` \| `feature_engineering`)
- `name` / description courte
- `type` (`stateless` — invariant garanti par le gate §6)
- `params` (dict des paramètres figés)
- **Implémentation** : si l'étape provient d'un script généré par l'agent, sauvegarder le **code source intégral** du script invoqué (chemin du fichier versionné + contenu)
- `invariance_test` : résultat du gate §6 (statut `passed`, vérifications effectuées, seeds des tirages de sous-échantillons) — une étape sans gate `passed` ne peut pas figurer dans l'historique
- `input_schema` / `output_schema` (colonnes + dtypes avant/après)
- horodatage, hash des données avant/après (optionnel mais recommandé pour l'audit)

Format suggéré : un `transformations.json` (métadonnées + références) accompagné d'un dossier `transforms/` contenant chaque script `.py` versionné. L'ensemble doit permettre de reconstruire la pipeline sans l'agent.

## 8. Contrats et invariants entre phases

Validation systématique entre chaque phase (Pydantic v2, complété par pandera si pertinent pour les invariants au niveau dataframe). Si un invariant échoue → arrêt explicite avec message clair (fail fast).

Invariants minimaux :

- **Entrée → Cleaning** : `target_column` présente ; CSV lisible ; schéma capturé.
- **Sortie Cleaning** : dtypes cohérents ; pas de doublons résiduels non intentionnels ; `target_column` intacte ; aucune transformation stateful enregistrée dans l'historique ; **toute étape porte un gate d'invariance §6 `passed`**.
- **Sortie Feature engineering** : `target_column` toujours présente et inchangée ; toutes les étapes marquées `stateless` avec gate §6 `passed` ; aucune colonne dérivée de la target ; pas de NaN introduit silencieusement (ou documenté).
- **Sortie Split** : trois jeux disjoints ; union = dataset post-FE ; proportions respectées ; (classification) distribution de la target préservée par stratification.

## 9. Phase 3 — Data splitting (déterministe)

- Réalisé par la classe `DataSplitter` dans `src/agentic_ml/data_manager/prepare_data.py`.
- **Tous** les paramètres de split sont **déterministes** et stockés dans `src/agentic_ml/data_manager/prepare_data.py` (ratios train/val/test, `random_state`/seed, flag de stratification).
- `task_type == classification` → split **stratifié** sur `target_column`.
- `task_type == regression` → split aléatoire seedé (optionnellement stratifié par bins de la target *au sein du split uniquement*, sans réinjecter de statistique dans les features).
- Aucune normalisation ni transformation fittée n'est appliquée ici : le split ne fait que partitionner.

## 10. Artefacts de sortie

Dans `output_dir` :

- `train.csv`, `val.csv`, `test.csv`
- `dtypes.json` : mapping colonne → dtype final, écrit à côté des CSV. Le CSV ne stocke pas les types ; ce fichier permet de les réimposer au rechargement (`pd.read_csv(..., dtype=...)` ou cast post-lecture) pour ne pas perdre le travail de typage de la phase cleaning.
- `transformations.json` + `transforms/*.py` (recette rejouable, §7)
- `metadata.json` : seed, ratios, hash du CSV source, versions des libs, schéma final, `target_column`, `task_type`, horodatage
- (optionnel) `schema.json` : schéma final validé (pandera/Pydantic)

## 11. Notes d'implémentation

- Utiliser des chemins POSIX (`/`) ou `pathlib.Path` partout (pas de `\` Windows).
- Phases strictement séquentielles ; pas de phase 2 avant phase 1 validée, etc.
- La frontière post-split (normalisation, encodages fittés) est **hors périmètre** de ce système mais l'historique doit être conçu pour s'y chaîner proprement.
- Prévoir un mode « dry run » qui produit la recette sans muter les données, utile pour audit du leakage avant exécution.

## 12. Critères d'acceptation

- [ ] **Chaque transformation des phases 1–2 a réussi le gate d'invariance par ligne (§6) avant d'être appliquée ; aucune étape n'est enregistrée sans gate `passed`.**
- [ ] Aucune transformation stateful/fitted n'apparaît dans l'historique des phases 1–2.
- [ ] La `target_column` n'est ni transformée ni source d'une feature.
- [ ] La recette (`transformations.json` + scripts) rejoue à l'identique sur un nouveau CSV de même schéma.
- [ ] Le split est bit-à-bit reproductible (même seed → mêmes partitions).
- [ ] Tous les contrats inter-phases passent ; un invariant violé arrête le pipeline avec un message explicite.
