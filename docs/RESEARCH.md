# Research log

This log lists the sources that we reviewed for the design of Meldr. It groups them by technique.
Each line gives the takeaway and what we did with it: **used**, **planned**, or **not used**
(with an explanation). `DECISIONS.md` records the decisions that follow from these sources.

## Confidence, calibration, and uncertainty

1. Dong, Halevy, Yu — *Data Integration with Uncertainty* (VLDB 2007). Match scores are rankings, not probabilities. Calibration is necessary for probabilistic mappings. **Used:** a calibrated threshold (`eval/calibration.py`).
2. *Calibrated confidence for LLM tabular QA* (2026). Verbalized confidence is badly calibrated on tables (ECE 0.35–0.64). Perturbation/consistency methods reach AUROC 0.78–0.86. **Used:** an opt-in self-consistency vote (`mapping.self_consistency`). Meldr blends the LLM confidence with the candidate and dry-run scores.
3. *Conformalized answer sets for KG embeddings* (2024). Conformal prediction gives coverage guarantees when the data is exchangeable. **Not used:** the exchangeability assumption is rarely true across new sources. Wilson bounds are simpler.
4. *Conformal calibration*, arXiv 2505.15437 (2025). It is from the same family and has set-valued outputs. **Not used**, as above.
5. CondKGCP (2025). Conditional conformal prediction. **Not used**, as above.
6. *Uncertainty in aggregate queries over integrated datasets* (2023). Mapping uncertainty propagates into downstream aggregates. **Used as a principle:** each run reports the values lost and the quarantine counts.
7. ConfTuner (NeurIPS 2025). It fine-tunes LLMs to verbalize calibrated confidence. **Not used:** model training is necessary, and we support hosted models.
8. LACIE (NeurIPS 2024). It fine-tunes LLMs for listener-aware calibration. **Not used**, as above.
9. *Extractive rationales and confidence* (GEM 2025). Rationale quality correlates with correctness. **Used in part:** every field keeps its rationale for reviewers.
10. ADVICE (2025). Confidence elicitation for LLM answers. **Not used:** it is a prompt-level fix, and sampling agreement is more robust.

## Human review and active learning

11. Splink documentation — the Fellegi-Sunter model, m/u probabilities, and thresholds. **Used:** `meldr eval` recommends a threshold from labeled precision. The configured threshold stays a constant that you set.
12. Robin Linacre — *The Fellegi-Sunter model* (blog). Match weights are log-odds, and you must calibrate them against labels. **Used**, as above.
13. *Active learning with Splink for biomedical record linkage* (2023). Label the most uncertain pairs first. **Used:** a review order by margin sampling (`pending_fields()`).
14. Splink discussion #2836 — threshold choice and clerical review bands. **Used:** when the labels cannot prove a threshold, the eval report tells you to continue to review everything.
15. Data Tamer (CIDR 2013). The machine proposes, experts approve, and the approvals improve subsequent proposals. **Used:** `ApprovedMappingMatcher`.
16. Tamr docs — mapping recommendations. Tamr gives ranked suggestions with a confidence for each attribute. **Used:** top-5 `candidates` in each spec.
17. Tamr blog — ML schema mapping at scale. Learning from curator feedback beats rules. **Used**, as above.
18. Snorkel (VLDB 2018 / VLDB Journal). Label models combine noisy signals without ground truth. **Not used at this time:** many labeled sources are necessary to beat fixed weights.
19. Stanford SAIL — weak supervision blog. It describes the same approach. **Not used at this time**, as above.
20. Snorkel `LabelModel` source code. We reviewed it to find the cost to add it. **Not used at this time.**

## Schema matching: benchmarks and metrics

21. Valentine (ICDE 2021). The standard benchmark for tabular schema matching. **Used:** `--suite tables`.
22. *Valentine in Action* (VLDB 2021 demo). An evaluation workflow. **Used.**
23. Valentine metrics documentation — recall@ground-truth and MRR-style ranking metrics. **Used:** the eval reports MRR and HitRate@1/3/5, and CI gates MRR and HitRate@1.
24. Magneto (VLDB 2025). Small-model retrieval, then LLM reranking. **Used:** candidate retrieval, then LLM adjudication.
25. Magneto retriever model card (Hugging Face). A fine-tuned retriever. **Not used:** it is a heavy extra, but the `Matcher` hook lets you add it subsequently.
26. Magneto synthetic training dataset. LLM-generated training pairs. **Not used:** we learn from approved mappings instead.
27. Parciak and others — *Schema matching with LLMs* (VLDB workshop 2024). Context and sampling strategies are important, and majority votes help. **Used:** self-consistency.
28. *Towards Scalable Schema Mapping with LLMs* (2025). If you batch the requests and prune the candidates, the cost stays linear. **Used:** chunked adjudication on shortlists.
29. *Automatic End-to-End Data Integration with LLMs* (PyDI team, 2026). Full pipelines are possible. The evaluation must cover values, not only matches. **Used:** value accuracy in the synthetic suite.
30. Starmie (VLDB 2023). Contrastive column embeddings for table discovery. **Not used:** discovery across data lakes is out of scope.
31. Megagon Labs — Starmie blog. It describes the same approach. **Not used.**

## Column types and entity matching

32. Sherlock (KDD 2019). Column-type detection from value features. **Used in part:** value-pattern features in the profiler.
33. Sato (VLDB 2020). It adds table context to Sherlock. **Not used:** it predicts DBpedia types, not user schemas.
34. Doduo (SIGMOD 2022). Transformer column annotation. **Not used**, as above.
35. *Evaluating column-type annotation models and benchmarks*. Benchmark leakage inflates results. **Used:** we test on held-out pairs of actual tables, not only synthetic data.
36. Ditto (VLDB 2021). Pre-trained language models for entity matching. **Not used:** you must fine-tune it, and our resolver stays deterministic.
37. *Deep entity matching: challenges* (JDIQ 2021). Deep models must have labeled pairs for each domain. **Not used**, as above.
38. *Evaluation of deep learning techniques for entity matching* (2023). The gains are smaller on actual dirty data. **Informs** the choice to keep blocking-plus-rules.
39. Peeters and others — *Fine-tuning LLMs for entity matching* (2024). Fine-tuning helps small models but rarely helps large ones. **Not used:** we use hosted models with prompts only.
40. Peeters and others — *Entity matching using LLMs* (EDBT 2025). Strong LLMs match zero-shot, and structured prompts help. **Used:** structured output with repair.
41. Jellyfish (EMNLP 2024). An instruction-tuned LLM for data preprocessing tasks. **Supported:** it runs through the OpenAI-compatible provider.
42. Unicorn (SIGMOD 2023). One model for many matching tasks. **Not used:** it is heavy, but the `Matcher` hook lets you add it.

## Clinical and scientific harmonization

43. *GPT-4 vs Usagi for OMOP concept mapping* (JMIR Medical Informatics 2025). LLMs rival specialist tools, but human review is necessary. **Used:** a review workflow with PII and always-review rules.
44. *Sentence-transformer mapping of EHR terms to OMOP*. Embeddings work well for retrieval. **Used:** embedding signal plus value shortlists.
45. Jackalope Plus (Scientific Reports 2025). Automated SNOMED post-coordination. **Informs** value mapping for large vocabularies.
46. *Multi-agent LLM OMOP mapping* (OHDSI 2025). Specialized agents plus verification. **Used in part:** a separate dry-run verification pass.
47. *LLM OMOP mapping tool* (JMIR 2025). Human confirmation stays mandatory in clinical settings. **Used.**
48. Harmony tool. Questionnaire-item harmonization with embeddings. **Informs** the embedding signal.
49. *Harmony validation* (BMC Psychiatry 2024). Embedding similarity agrees with human experts. **Informs**, as above.

## Data cleaning and transformation

50. Baran (VLDB 2020). Learned error correction. **Not used:** Meldr quarantines bad values and does not rewrite them.
51. Raha and Baran (CIDR 2021). Configuration-free error detection. **Used in part:** sentinel and pattern detection in the profiler.
52. HoloDetect (SIGMOD 2019). Few-shot error detection. **Not used:** training is necessary.
53. DaskRahaBaran. Distributed error detection. **Not used.**
54. Cocoon (2024). It uses an LLM to clean tables semantically. **Used as a principle:** the LLM proposes value maps, and Meldr applies them deterministically.
55. AutoDCWorkflow (EMNLP Findings 2025). LLMs make workflows from a fixed set of operations to clean data. **Used:** the whitelisted `TransformRegistry` DSL.
56. Tian Pan — *LLM data migrations* (blog, 2026). Make the migration one time, make sure that it is correct, and then run it deterministically. **Used:** the core architecture.
57. Databricks — *SAP semantic metadata* (blog). Business metadata improves automated mapping. **Used:** synonyms, descriptions, and examples on canonical fields.
58. Databricks — *Lakehouse Federation* (blog). Query the data in place, and standardize the contracts. **Informs** the connector design.

## Libraries, standards, and earlier reviews

59. BDI-Kit ([github.com/VIDA-NYU/bdi-kit](https://github.com/VIDA-NYU/bdi-kit), Patterns 2026, demo paper). A composable matching API. **Used:** `propose_mapping` / `apply_mapping`.
60. Harmonia agent, NYU VIDA (arXiv 2502.07132). An LLM agent for interactive harmonization. **Naming conflict:** resolved. The name of the library is Meldr.
61. SCHEMORA. LLM metadata enrichment before retrieval. **Not used:** it makes calls before a cache hit can occur.
62. Matchmaker (ICML 2025). Self-improving compositional matcher. **Used in part:** learning from approvals.
63. ReMatch. Retrieval-augmented matching. **Used:** shortlist, then adjudicate.
64. KG-RAG4SM. Knowledge-graph retrieval for matching. **Not used:** an external knowledge graph is necessary.
65. *LLM schema matching*, arXiv 2505.24716. Self-consistency improves precision. **Used.**
66. schema-crosswalk. Versioned transform semantics. **Used:** `TRANSFORMS_VERSION` pin.
67. PyDI. Python data integration toolkit. Compared the API surface.
68. datastew. Embedding-based variable mapping. Compared.
69. schemashift. LLM schema migration. Compared.
70. ddharmon. Data dictionary harmonization. Compared.
71. dlt documentation. Incremental loads and schema evolution. **Informs** idempotent upserts and drift.
72. LinkML ([linkml.io](https://linkml.io)). Schema language used in biomedical harmonization. **Used:** LinkML import.
73. LinkML-Map. Declarative transformation specs. **Planned:** compile specs to it.
74. SSSOM specification and schema ([mapping-commons.github.io/sssom](https://mapping-commons.github.io/sssom/)). Standard mapping exchange format. **Used:** SSSOM/TSV export.
75. TabulaX (VLDB 2025). It classifies value transforms (string, numeric, algorithmic, general). **Used:** op coverage check, `scale(offset)`.
76. MegaTran (VLDB 2025). LLM-driven value transformation. **Used as a principle:** the LLM proposes, and the DSL runs the transformation.
77. *Programming by example meets LLMs* (2025). PBE plus LLMs for transforms. **Not used:** this method makes programs that run as code.
78. MMTU (NeurIPS 2025). A broad benchmark that tests how well models understand tables. **Informs** the evaluation scope.
79. Valentine NYU dataset ([github.com/delftdata/valentine](https://github.com/delftdata/valentine)). Table pairs from actual data, with ground truth. **Used:** CI benchmark gate.
80. Arpit Bhayani — *Databases Were Not Designed For This* (Rootconf 2026 keynote, [blog](https://arpitbhayani.me/blogs/defensive-databases/)). Treat the data layer as defensive: brief transactions, tagged queries, idempotent writes, no silent failures, legible schemas, and least privilege. **Used:** staged SQL reads, query tags, quarantine of failed fields, tables that describe themselves, and locked-down sink queries.
