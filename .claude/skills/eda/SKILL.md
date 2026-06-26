---
name: eda
description: "Génère automatiquement un notebook d'analyse exploratoire (EDA) pour un dataset CSV, en partant du template notebooks/data_analysis_template.ipynb. Détecte automatiquement les colonnes, le type de tâche et propose une conclusion."
---

# /eda

Génère un notebook d'analyse exploratoire complet pour un dataset, basé sur `notebooks/data_analysis_template.ipynb`.

## Syntaxe

```
/eda <chemin_dataset>
/eda <chemin_dataset> --target <colonne_cible>
/eda <chemin_dataset> --target <colonne_cible> --task classification|regression
/eda <chemin_dataset> --target <colonne_cible> --task classification --drop col1,col2
```

**Arguments :**
- `<chemin_dataset>` — chemin vers le CSV (relatif à la racine du projet, ex: `data/00_raw/titanic.csv`)
- `--target` — colonne cible (si omis : détectée automatiquement, voir ci-dessous)
- `--task` — `classification` ou `regression` (si omis : inféré depuis le contenu de la target)
- `--drop` — colonnes à exclure, séparées par des virgules (ex: `--drop Id,PassengerId`)

## Ce que tu dois faire quand ce skill est invoqué

### Étape 1 — Parser les arguments

Extraire depuis les args :
- `dataset_path` : le premier argument positionnel
- `target_col` : valeur de `--target` si présent, sinon `None`
- `task` : valeur de `--task` si présent, sinon `None`
- `drop_cols` : liste depuis `--drop col1,col2` si présent, sinon `[]`

### Étape 2 — Inspecter le dataset

Lire les **20 premières lignes** du CSV avec le Read tool pour obtenir :
1. La liste des colonnes
2. Les types de données apparents (nombres, texte, catégoriel)

**Auto-détection si `--target` non fourni :**
- Cherche une colonne nommée `target`, `label`, `class`, `Species`, `y`, `outcome`, `survived`, `price`, `salary` (insensible à la casse)
- Sinon, prends la **dernière colonne** du CSV
- Annonce à l'utilisateur quelle colonne a été choisie

**Auto-détection du task si `--task` non fourni :**
- Si la target a ≤ 20 valeurs uniques distinctes (ou est de type string/object) → `classification`
- Sinon → `regression`
- Annonce le choix à l'utilisateur

**Auto-détection des colonnes à dropper :**
- Propose de dropper les colonnes qui ressemblent à des IDs (nommées `id`, `Id`, `ID`, `index`, `Index`, `row_id`, etc.)
- Si `--drop` n'est pas fourni et qu'il y a des colonnes ID candidates, ajoute-les automatiquement à `drop_cols`

### Étape 3 — Calculer le nom du notebook de sortie

```
notebook_name = "data_analysis_" + <nom_du_fichier_sans_extension>.lower() + ".ipynb"
output_path   = "notebooks/" + notebook_name
```

Exemple : `data/00_raw/Titanic.csv` → `notebooks/data_analysis_titanic.ipynb`

### Étape 4 — Lire le template

Lis `notebooks/data_analysis_template.ipynb` avec le Read tool.

### Étape 5 — Générer le notebook

Crée `notebooks/<notebook_name>` en partant du JSON du template et en faisant les substitutions suivantes :

**Cellule markdown titre (cell index 0) :**
```
# Analyse exploratoire — [DATASET NAME]
```
→ Remplace `[DATASET NAME]` par le nom du fichier sans extension (ex: `Titanic`)

**Cellule de configuration (cell index 2, la cellule "## 2. Configuration") :**
Remplace le contenu par :
```python
# ── À adapter pour chaque dataset ──────────────────────────────────────────
DATA_PATH   = "<chemin_relatif_depuis_notebooks>"
TARGET_COL  = "<target_col>"
TASK        = "<task>"
DROP_COLS   = <drop_cols_as_python_list>
# ───────────────────────────────────────────────────────────────────────────
```

Le `DATA_PATH` doit être relatif au dossier `notebooks/`, donc préfixé par `../` si le dataset est dans `data/`.

Exemples de valeurs finales :
```python
DATA_PATH   = "../data/00_raw/Titanic.csv"
TARGET_COL  = "Survived"
TASK        = "classification"
DROP_COLS   = ["PassengerId", "Name"]
```

Toutes les autres cellules restent **identiques au template**.

Écris le fichier avec le Write tool.

### Étape 6 — Confirmer

Affiche un résumé :
```
Notebook créé : notebooks/data_analysis_<name>.ipynb
  Dataset  : <dataset_path>
  Target   : <target_col> (<task>)
  Drop     : <drop_cols>
```

Si des choix ont été faits automatiquement (target, task, drop), indique-le clairement pour que l'utilisateur puisse corriger si besoin.
