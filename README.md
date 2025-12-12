# Graph Autoencoder (GAE) Anomaly Detection

A Graph Autoencoder-based anomaly detection system with rare pattern mining for enhanced detection performance. The pipeline constructs k-NN graphs from behavioral features, trains a GAE model, and enhances detection using the Apriori algorithm.

**Best Performance:** nDCG score of 0.66 with rare pattern boosting

## Requirements

Python 3.8+ required. Install dependencies using:

```bash
pip install -r requirements.txt
```

Or create a conda environment:

```bash
conda env create -f environment.yml
conda activate anomaly
```

## Pipeline

### 1. Data Exploration

```bash
jupyter notebook 0_data_cleaning.ipynb
```

**Purpose:** Analyze dataset characteristics, feature distributions, and data quality

### 2. Graph Construction

```bash
jupyter notebook 1_graph_construction.ipynb
```

**Purpose:** Build k-NN graphs from feature vectors

**Key Parameters:**

- `K_NEIGHBORS = 5` - Number of nearest neighbors
- `METRIC = 'cosine'` - Distance metric

**Outputs:**

- `data/train_graph.pt` - Training graph with normalized features
- `data/test_graph.pt` - Test graph with normalized features

### 3. Model Training

```bash
jupyter notebook 2_train_gae.ipynb
```

**Purpose:** Train Graph Autoencoder for link prediction

**Key Parameters:**

- `HIDDEN_DIM = 128` - Hidden layer dimension
- `LATENT_DIM = 64` - Embedding dimension
- `NUM_EPOCHS = 200` - Training epochs
- `LEARNING_RATE = 0.05` - Learning rate
- `WEIGHT_DECAY = 0` - L2 regularization

**Outputs:**

- `models/gae_trained.pt` - Trained model checkpoint

**Expected Results:**

- Final loss: ~0.76

### 4. Rare Pattern Mining

```bash
jupyter notebook 3_rare_pattern_mining.ipynb
```

**Purpose:** Mine rare behavioral patterns for score boosting

**Key Parameters:**

- `MAX_SUPPORT = 0.01` - Maximum support threshold (1%)
- `MIN_CONFIDENCE = 0.7` - Minimum confidence for association rules

**Outputs:**
- `rare_patterns/rare_graph_test.pt` - Rare pattern graph

### 5. Evaluation

```bash
jupyter notebook 4_evaluation.ipynb
```

**Purpose:** Evaluate model performance with and without rare pattern boosting

**Key Parameters:**

- `ALPHA` values - Boosting weight

**Expected Results:**

- Base model nDCG: ~0.52
- With rare pattern boosting (α=2): ~0.66

**Outputs:**

- nDCG scores for different α values

## Project Structure

```
parent_directory/
├── GAE/
│   ├── 0_data_cleaning.ipynb          # Data exploration
│   ├── 1_graph_construction.ipynb     # Build k-NN graphs
│   ├── 2_train_gae.ipynb              # Train GAE model
│   ├── 3_rare_pattern_mining.ipynb    # Mine rare patterns
│   ├── 4_evaluation.ipynb             # Evaluate performance
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── models.py                  # GAE model definitions
│   │   ├── training.py                # Training utils
│   │   ├── evaluation.py              # Evaluation metrics utils
│   │   ├── graph_construction.py      # k-NN graph building utils
│   │   ├── file_utils.py              # Save/load utilities
│   │   ├── apriori.py                 # Apriori algorithm
│   │   └── rare_patterns.py           # Rare pattern utilities
│   ├── data/                          # Generated graphs
│   ├── models/                        # Trained models
│   └── rare_patterns/                 # Rare pattern graph
└── e2-master/                         # The dataset

```
