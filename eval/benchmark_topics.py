"""
eval/benchmark_topics.py
────────────────────────
Ground-truth benchmark definitions for the three well-studied topics
used to evaluate the Literature Analysis System.

Each topic entry contains:
  - canonical_papers  : the must-find papers for recall measurement.
                        Titles are lowercase-normalised so matching is
                        case-insensitive.
  - expected_clusters : methodology clusters the Analyst should produce.
  - expected_contradictions : known real tensions in the literature.
  - paradigm_shifts   : key historical inflection points to verify against
                        the Historian pass output.

Usage
-----
    from eval.benchmark_topics import BENCHMARK_TOPICS
    topic = BENCHMARK_TOPICS["attention_mechanisms"]

These are consumed by eval/metrics.py.  Do not change titles after the
benchmark has been run — treat them as a frozen test fixture.
"""

from typing import TypedDict


class CanonicalPaper(TypedDict):
    title_lower: str          # lowercase, used for fuzzy matching
    year: int
    authors_hint: str         # first author surname, helps resolve ambiguity
    importance: str           # "core" | "supporting"


class BenchmarkTopic(TypedDict):
    display_name: str
    query: str                # the exact query string to feed the Retriever
    canonical_papers: list[CanonicalPaper]
    expected_clusters: list[str]
    expected_contradictions: list[str]   # plain-English descriptions
    paradigm_shifts: list[dict]          # {year, description}


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC 1 — Attention Mechanisms in NLP
# ─────────────────────────────────────────────────────────────────────────────
_ATTENTION: BenchmarkTopic = {
    "display_name": "Attention Mechanisms in NLP",
    "query": "attention mechanisms natural language processing",
    "canonical_papers": [
        # Core papers — system MUST retrieve these for a passing recall score
        {"title_lower": "attention is all you need",
         "year": 2017, "authors_hint": "vaswani", "importance": "core"},
        {"title_lower": "neural machine translation by jointly learning to align and translate",
         "year": 2015, "authors_hint": "bahdanau", "importance": "core"},
        {"title_lower": "bert: pre-training of deep bidirectional transformers for language understanding",
         "year": 2019, "authors_hint": "devlin", "importance": "core"},
        {"title_lower": "language models are few-shot learners",
         "year": 2020, "authors_hint": "brown", "importance": "core"},
        {"title_lower": "exploring the limits of transfer learning with a unified text-to-text transformer",
         "year": 2020, "authors_hint": "raffel", "importance": "core"},
        # Supporting papers — important but not strictly required
        {"title_lower": "long short-term memory",
         "year": 1997, "authors_hint": "hochreiter", "importance": "supporting"},
        {"title_lower": "sequence to sequence learning with neural networks",
         "year": 2014, "authors_hint": "sutskever", "importance": "supporting"},
        {"title_lower": "effective approaches to attention-based neural machine translation",
         "year": 2015, "authors_hint": "luong", "importance": "supporting"},
        {"title_lower": "an image is worth 16x16 words: transformers for image recognition at scale",
         "year": 2021, "authors_hint": "dosovitskiy", "importance": "supporting"},
        {"title_lower": "roberta: a robustly optimized bert pretraining approach",
         "year": 2019, "authors_hint": "liu", "importance": "supporting"},
    ],
    "expected_clusters": [
        "transformer-based",
        "RNN-based",
        "survey",
    ],
    "expected_contradictions": [
        "RNN-based models (Bahdanau) vs transformer-based models (Vaswani) on sequence modelling efficiency",
        "Fixed-length encoder bottleneck (Sutskever) vs dynamic alignment (Bahdanau)",
        "Absolute positional encoding (Vaswani) vs relative positional encoding (Shaw et al.)",
    ],
    "paradigm_shifts": [
        {"year": 2015, "description": "Bahdanau attention breaks the fixed-length bottleneck in encoder-decoder NMT"},
        {"year": 2017, "description": "Transformer replaces recurrence entirely — attention is all you need"},
        {"year": 2018, "description": "ELMo / BERT introduce contextualised pre-training, making task-specific architectures obsolete"},
        {"year": 2020, "description": "GPT-3 demonstrates few-shot learning at scale, shifting focus to prompt engineering"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC 2 — Reinforcement Learning from Human Feedback (RLHF)
# ─────────────────────────────────────────────────────────────────────────────
_RLHF: BenchmarkTopic = {
    "display_name": "Reinforcement Learning from Human Feedback (RLHF)",
    "query": "reinforcement learning from human feedback language model alignment",
    "canonical_papers": [
        {"title_lower": "training language models to follow instructions with human feedback",
         "year": 2022, "authors_hint": "ouyang", "importance": "core"},
        {"title_lower": "learning to summarize from human feedback",
         "year": 2022, "authors_hint": "stiennon", "importance": "core"},
        {"title_lower": "deep reinforcement learning from human preferences",
         "year": 2017, "authors_hint": "christiano", "importance": "core"},
        {"title_lower": "constitutional ai: harmlessness from ai feedback",
         "year": 2022, "authors_hint": "bai", "importance": "core"},
        {"title_lower": "direct preference optimization: your language model is secretly a reward model",
         "year": 2023, "authors_hint": "rafailov", "importance": "core"},
        {"title_lower": "proximal policy optimization algorithms",
         "year": 2017, "authors_hint": "schulman", "importance": "supporting"},
        {"title_lower": "fine-tuning language models from human preferences",
         "year": 2020, "authors_hint": "ziegler", "importance": "supporting"},
        {"title_lower": "reward model ensembles help mitigate overoptimization",
         "year": 2023, "authors_hint": "coste", "importance": "supporting"},
        {"title_lower": "scaling laws for reward model overoptimization",
         "year": 2022, "authors_hint": "gao", "importance": "supporting"},
    ],
    "expected_clusters": [
        "reinforcement-learning",
        "transformer-based",
        "optimization",
    ],
    "expected_contradictions": [
        "PPO-based RLHF (Ouyang/InstructGPT) vs DPO (Rafailov) on whether a separate RL step is necessary",
        "Reward model reliability vs reward hacking / overoptimisation (Gao et al. scaling laws)",
        "RLHF requiring large human preference datasets vs Constitutional AI using AI feedback to reduce human annotation",
    ],
    "paradigm_shifts": [
        {"year": 2017, "description": "Christiano et al. show RL from human preferences can train policies without hand-crafted reward functions"},
        {"year": 2020, "description": "Ziegler et al. apply RLHF to language model fine-tuning for the first time"},
        {"year": 2022, "description": "InstructGPT / ChatGPT demonstrate RLHF at scale produces dramatically more helpful LLMs"},
        {"year": 2023, "description": "DPO bypasses the RL training loop entirely — directly optimises on preferences"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC 3 — Graph Neural Networks (GNNs)
# ─────────────────────────────────────────────────────────────────────────────
_GNN: BenchmarkTopic = {
    "display_name": "Graph Neural Networks",
    "query": "graph neural networks node classification link prediction",
    "canonical_papers": [
        {"title_lower": "semi-supervised classification with graph convolutional networks",
         "year": 2017, "authors_hint": "kipf", "importance": "core"},
        {"title_lower": "how powerful are graph neural networks?",
         "year": 2019, "authors_hint": "xu", "importance": "core"},
        {"title_lower": "graph attention networks",
         "year": 2018, "authors_hint": "velickovic", "importance": "core"},
        {"title_lower": "inductive representation learning on large graphs",
         "year": 2017, "authors_hint": "hamilton", "importance": "core"},
        {"title_lower": "convolutional neural networks on graphs with fast localized spectral filtering",
         "year": 2016, "authors_hint": "defferrard", "importance": "core"},
        {"title_lower": "the graph neural network model",
         "year": 2009, "authors_hint": "scarselli", "importance": "supporting"},
        {"title_lower": "deep graph infomax",
         "year": 2019, "authors_hint": "velickovic", "importance": "supporting"},
        {"title_lower": "predict then propagate: graph neural networks meet personalized pagerank",
         "year": 2019, "authors_hint": "gasteiger", "importance": "supporting"},
        {"title_lower": "dropout: a simple way to prevent neural networks from overfitting",
         "year": 2014, "authors_hint": "srivastava", "importance": "supporting"},
        {"title_lower": "graph transformer networks",
         "year": 2019, "authors_hint": "yun", "importance": "supporting"},
    ],
    "expected_clusters": [
        "GNN-based",
        "transformer-based",
        "optimization",
    ],
    "expected_contradictions": [
        "Spectral methods (Kipf GCN) vs spatial/inductive methods (Hamilton GraphSAGE) on generalisation to unseen nodes",
        "Fixed neighbourhood aggregation (GCN) vs learned attention weights over neighbours (GAT)",
        "GNN expressiveness bounded by 1-WL test (Xu et al.) vs claims of superior performance in practice",
    ],
    "paradigm_shifts": [
        {"year": 2009, "description": "Scarselli et al. formally define the Graph Neural Network model — first principled graph deep learning framework"},
        {"year": 2016, "description": "ChebNet / GCN bring convolutional ideas to graphs via spectral graph theory"},
        {"year": 2017, "description": "GraphSAGE enables inductive learning on unseen nodes — removes the transductive limitation of GCN"},
        {"year": 2018, "description": "Graph Attention Networks introduce dynamic, learned edge weights replacing fixed normalisation"},
        {"year": 2019, "description": "Xu et al. theoretically characterise GNN expressiveness — WL isomorphism test sets the ceiling"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Public registry — import this in metrics.py
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_TOPICS: dict[str, BenchmarkTopic] = {
    "attention_mechanisms": _ATTENTION,
    "rlhf":                 _RLHF,
    "graph_neural_networks": _GNN,
}

# Convenience alias: list of topic keys in canonical evaluation order
TOPIC_KEYS: list[str] = list(BENCHMARK_TOPICS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test — run directly to verify the data is internally consistent
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for key, topic in BENCHMARK_TOPICS.items():
        core_count = sum(1 for p in topic["canonical_papers"] if p["importance"] == "core")
        total_count = len(topic["canonical_papers"])
        print(f"\n{'─'*60}")
        print(f"Topic key  : {key}")
        print(f"Display    : {topic['display_name']}")
        print(f"Query      : {topic['query']}")
        print(f"Papers     : {total_count} total ({core_count} core, {total_count - core_count} supporting)")
        print(f"Clusters   : {topic['expected_clusters']}")
        print(f"Contradictions : {len(topic['expected_contradictions'])}")
        print(f"Paradigm shifts: {len(topic['paradigm_shifts'])}")
    print(f"\n{'─'*60}")
    print(f"All {len(BENCHMARK_TOPICS)} topics loaded OK.")
