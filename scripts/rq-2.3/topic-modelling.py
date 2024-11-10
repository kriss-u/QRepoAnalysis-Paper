import collections

import hdbscan
import nltk
import numpy as np
import torch
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

nltk.download("punkt")
torch.cuda.set_device(0)
print(torch.cuda.current_device())


def save_embeddings(docs):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True)

    with open("embeddings.npy", "wb") as f:
        np.save(f, embeddings)
    return


def load_embeddings():
    embeddings = np.load("embeddings.npy")
    return embeddings


def prepare_vocab(docs):
    vocab_counter = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(docs):
        vocab_counter.update(tokenizer(doc))

    vocab = [word for word, frequency in vocab_counter.items() if frequency >= 20]  ##
    return vocab


def get_aspects(docs, embeddings, vocab, seed_topic_list, cidx):
    print("fitting")
    # Prepare sub-models
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(
        n_components=10,
        n_neighbors=20,
        random_state=42,
        metric="cosine",
        min_dist=0.1,
        low_memory=True,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_samples=20,
        gen_min_span_tree=True,
        prediction_data=False,
        min_cluster_size=10,
    )
    vectorizer_model = CountVectorizer(
        vocabulary=vocab, stop_words="english", ngram_range=(1, 3), min_df=10
    )
    cluster_model = KMeans(n_clusters=cidx)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        top_n_words=20,
        n_gram_range=(1, 3),
        seed_topic_list=seed_topic_list,
    ).fit(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()

    topic_model.save(
        path=f"final-{cidx}",
        serialization="safetensors",
        save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


if __name__ == "__main__":

    sentences = torch.load("sentences.pt")
    print(len(sentences))
    sentences = [item.lower() for item in sentences if item]
    sentences = list(set(sentences))
    print(len(sentences))
    print()
    print()

    seed_topic_list = [
        [
            "bump",
            "update",
            "version",
            "release",
            "Requirement",
            "dependency",
            "upgrade",
            "deprecation",
            "deprecate",
            "pypi",
            "pip",
            "install",
            "installation",
        ],
        [
            "fix",
            "remove",
            "error",
            "fail",
            "bug",
            "issue",
            "check",
            "run",
            "warning",
            "incorrect",
            "bugfix",
            "crash",
            "fixed",
            "failure",
            "wrong",
            "avoid",
            "log",
            "compatibility",
            "Patch",
            "exception",
        ],
        [
            "support",
            "change",
            "improve",
            "improvement",
            "refactor",
            "rename",
            "implementation",
            "feature",
            "optimization",
            "cleanup",
            "clean",
            "optimize",
            "simplify",
        ],
        [
            "gate",
            "circuit",
            "qubit",
            "quantum",
            "measurement",
            "pulse",
            "matrix",
            "simulator",
            "decomposition",
            "phase",
            "hamiltonian",
            "algorithm",
            "datum",
            "classical",
            "simulation",
            "gradient",
            "pauli",
            "estimation",
            "rotation",
            "clifford",
            "measure",
        ],
        [
            "doc",
            "documentation",
            "example",
            "tutorial",
            "typo",
            "sphinx",
            "readme",
            "instruction",
            "docstring",
            "document",
            "docs",
            "latex",
        ],
        [
            "python",
            "qiskit",
            "numpy",
            "mypy",
            "rust",
            "pillow",
            "jax",
            "qutip",
            "json",
            "ruff",
            "cirq",
            "pyquil",
            "jupyter",
            "matplotlib",
            "tensorflow",
            "qasm",
            "scipy",
            "notebook",
            "ipython",
            "ipykernel",
            "docker",
            "pylint",
            "pennylane",
            "terra",
            "qobj",
            "qaoa",
            "aer",
            "openqasm",
        ],
        [
            "build",
            "ci",
            "workflow",
            "test",
            "testing",
            "coverage",
            "integration",
            "travis",
            "debug",
            "pytest",
        ],
        [
            "class",
            "import",
            "module",
            "return",
            "pass",
            "break",
            "raise",
            "try",
            "kwarg",
            "export",
            "seed",
            "decorator",
        ],
        ["performance", "experiment", "benchmark", "calibration"],
    ]

    save_embeddings(sentences)
    embeddings = load_embeddings()
    print("vocab")
    vocab = prepare_vocab(sentences)
    torch.save(vocab, "vocab.pt")
    print("get aspects")
    get_aspects(sentences, embeddings, vocab, seed_topic_list, 20)
