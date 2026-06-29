# Graph Report - agentic-ml  (2026-06-29)

## Corpus Check
- 43 files · ~15,607 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 347 nodes · 544 edges · 25 communities (21 shown, 4 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 62 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dc8d3022`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]

## God Nodes (most connected - your core abstractions)
1. `PrepState` - 16 edges
2. `_agent_step()` - 15 edges
3. `AgentState` - 14 edges
4. `Système agentique de préparation de données pour le ML` - 13 edges
5. `run_agent()` - 10 edges
6. `write_artifacts()` - 9 edges
7. `run_invariance_gate()` - 9 edges
8. `launch_ml_pipeline()` - 9 edges
9. `HyperparameterOptimizer` - 9 edges
10. `_assert()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `parse_args()` --calls--> `available_models()`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/models.py
- `main()` --calls--> `HyperparameterOptimizer`  [INFERRED]
  scripts/run_optimization.py → src/agentic_ml/training/optimizer.py
- `main()` --calls--> `run_agent()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/graph.py
- `main()` --calls--> `format_class_report()`  [INFERRED]
  scripts/run_training_agent.py → src/agentic_ml/agents/training_agent/report.py
- `test_launch_ml_pipeline_is_json_serializable()` --calls--> `launch_ml_pipeline()`  [INFERRED]
  tests/test_pipeline_serialization.py → src/agentic_ml/agents/training_agent/tools.py

## Import Cycles
- None detected.

## Communities (25 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (36): Path, Exécute la boucle agentique de bout en bout et renvoie l'état final.      Args:, run_agent(), evaluate_stop(), _evaluate_stop_agent(), _evaluate_stop_convergence(), propose_experiment(), Nœuds de la boucle agentique.  Seul `propose_experiment` mobilise le LLM. `run_p (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (24): État partagé de l'agent, réinjecté à chaque tour de la boucle LangGraph.  Reflèt, Un essai enregistré dans le `trial_log`., Trial, load_prepared_run(), PreparedData, Chargement d'un run de données préparées pour l'entraînement., Données d'un run, prêtes pour fit/predict., Charge train/val d'un run, sépare X/y et encode la cible en entiers.      L'enco (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.50
Nodes (4): _assert_class_report(), Test de non-régression : la sortie de `launch_ml_pipeline` doit être entièrement, Vérifie la présence, la forme et la sérialisabilité du rapport par classe., test_launch_ml_pipeline_is_json_serializable()

### Community 3 - "Community 3"
Cohesion: 0.40
Nodes (4): agentic-ml, Conventions, graphify, Structure

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (59): BaseModel, DataFrame, _assert(), _assert_target_intact(), check_cleaning_output(), check_entry(), check_fe_output(), check_split_output() (+51 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (15): Namespace, build_prepare_graph(), Assemblage du StateGraph de préparation et point d'entrée `run_prepare_agent`., Construit et compile le graphe du pipeline de préparation., Exécute le pipeline cleaning → FE → split de bout en bout.      `input_csv` et `, run_prepare_agent(), main(), parse_args() (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (41): Any, ExtraTreesClassifier, GaussianNB, HistGradientBoostingClassifier, LinearDiscriminantAnalysis, Pipeline, RandomForestClassifier, SVC (+33 more)

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (7): build_step_context(), cleaning_system_prompt(), fe_system_prompt(), _history_brief(), Prompts système (mission) et construction du contexte des deux agents.  Les list, Assemble le contexte dynamique injecté à l'agent pour le prochain pas., Vue compacte de l'historique (sans le code intégral, trop verbeux).

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (9): Experiment, HyperparameterRange, Sortie structurée du LLM pour une proposition d'essai.  Le LLM renvoie une liste, Plage d'un hyperparamètre à explorer., Proposition d'essai : hypothèse explicite + configuration., Décision d'arrêt prise par le LLM (mode d'arrêt « agent »)., Convertit la liste de plages en dict au format `search_space`., StopDecision (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.23
Nodes (11): apply_transform(), compile_transform(), ExecutorError, Exécution déterministe des scripts générés par l'agent (§5).  > L'agent décide ;, Le script est invalide, ne définit pas `transform`, ou lève à l'exécution., Compile le script et renvoie la fonction `transform`.      Lève `ExecutorError`, Applique `fn` à une **copie** de `df` et valide grossièrement la sortie.      Lè, Rejoue la recette d'un répertoire de sortie sur un dataframe `df`.      Les étap (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.20
Nodes (9): Ce que tu dois faire quand ce skill est invoqué, /eda, Syntaxe, Étape 1 — Parser les arguments, Étape 2 — Inspecter le dataset, Étape 3 — Calculer le nom du notebook de sortie, Étape 4 — Lire le template, Étape 5 — Générer le notebook (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.10
Nodes (20): 10. Artefacts de sortie, 11. Notes d'implémentation, 12. Critères d'acceptation, 1. Objectif, 2.1 Définition, 2.2 Liste explicite — AUTORISÉ (stateless), 2.3 Liste explicite — INTERDIT avant le split (stateful / fitted), 2. Principe fondamental : zéro fuite de données (data leakage) (+12 more)

### Community 13 - "Community 13"
Cohesion: 0.19
Nodes (11): _dtypes_map(), _lib_versions(), Écriture des artefacts de sortie (§10) dans `output_dir`.  - train/val/test.csv, Matérialise tous les artefacts et renvoie le répertoire de sortie., write_artifacts(), _write_json(), Recette rejouable (§7) : sérialisation et ré-application sans l'agent.  L'histor, Chemin relatif déterministe du script versionné d'une étape. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.09
Nodes (20): BaseCallbackHandler, BaseChatModel, LLMResult, make_llm(), Instancie le client LLM selon AGENT_PROVIDER, avec rate limiting., build_agent_graph(), Assemblage du StateGraph et point d'entrée `run_agent`., Construit et compile le graphe de la boucle de recherche.      Flux : propose_ex (+12 more)

### Community 22 - "Community 22"
Cohesion: 0.27
Nodes (4): DataSplitter, next_run_folder(), Renvoie le prochain dossier de run versionné `<dataset>_<idx>` (idx zfill)., Partitionne un dataframe **déjà en mémoire** (classification stratifiée).

### Community 23 - "Community 23"
Cohesion: 0.29
Nodes (7): hash_file(), profile_dataframe(), Fonctions utilitaires : profil du dataframe, capture de schéma, hachage.  Le pro, Rend une valeur JSON-sérialisable (types numpy → natifs)., Hash sha256 d'un fichier (provenance du CSV source §4)., Profil descriptif du dataframe pour le contexte de l'agent.      Marque la colon, _to_python()

### Community 24 - "Community 24"
Cohesion: 0.50
Nodes (3): Sortie structurée du LLM pour une étape de cleaning ou de feature engineering., Décision de l'agent pour le prochain pas d'une phase., StepProposal

## Knowledge Gaps
- **27 isolated node(s):** `Syntaxe`, `Étape 1 — Parser les arguments`, `Étape 2 — Inspecter le dataset`, `Étape 3 — Calculer le nom du notebook de sortie`, `Étape 4 — Lire le template` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_agent_step()` connect `Community 4` to `Community 7`, `Community 9`, `Community 13`, `Community 20`, `Community 23`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `build_estimator()` connect `Community 1` to `Community 0`, `Community 6`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `PrepState` connect `Community 4` to `Community 5`, `Community 13`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `_agent_step()` (e.g. with `apply_transform()` and `compile_transform()`) actually correct?**
  _`_agent_step()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Entry point: lance une optimisation d'hyperparamètres (Optuna TPE) sur un run pr`, `Entry point: run the agentic data-preparation pipeline (cleaning → FE → split).`, `Entry point: run the autonomous model/hyperparameter search agent.` to the rest of the system?**
  _148 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07564102564102564 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08669354838709678 - nodes in this community are weakly interconnected._