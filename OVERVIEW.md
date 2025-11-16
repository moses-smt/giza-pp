# GIZA++ and mkcls: Algorithm Overview

This document provides a comprehensive technical overview of the GIZA++ word alignment toolkit and the mkcls word clustering tool.

## Table of Contents

1. [Introduction](#introduction)
2. [GIZA++ Word Alignment Algorithm](#giza-word-alignment-algorithm)
3. [mkcls Word Clustering Algorithm](#mkcls-word-clustering-algorithm)
4. [References](#references)

---

## Introduction

This repository contains two key components for statistical machine translation:

- **GIZA++**: An implementation of the IBM alignment models (1-5) and HMM alignment model for learning word alignments from parallel bilingual text
- **mkcls**: A maximum-likelihood word clustering tool that groups words into classes to improve generalization in language and translation models

Both tools were originally developed by Franz Josef Och and colleagues, and have been foundational in the field of statistical machine translation.

---

## GIZA++ Word Alignment Algorithm

### Purpose and Background

GIZA++ learns statistical word alignment models from parallel bilingual text (bitex). Given sentence pairs in two languages, it discovers which words correspond to each other across the translation. These alignments are essential for:

- Training phrase-based and hierarchical machine translation systems
- Extracting translation lexicons
- Understanding translation phenomena

### Model Cascade

GIZA++ implements a cascade of progressively more sophisticated alignment models, typically trained in sequence:

1. **IBM Model 1** (5 iterations) - Basic lexical translation
2. **IBM Model 2** (optional) - Adds absolute position
3. **HMM Alignment Model** (5 iterations) - First-order positional dependencies
4. **IBM Model 3** (5 iterations) - Fertility-based alignment
5. **IBM Model 4** (5 iterations) - Relative positioning with word classes
6. **IBM Model 5** (optional) - Vacancy-based distortion

Each model initializes from the previous one, enabling progressive refinement of alignment quality.

### Core Data Structures

#### Translation Tables (TTables)
- **Purpose**: Store translation probabilities P(f|e)
- **Implementation**: Hash map from word pairs to probability values
- **Location**: `GIZA++-v2/TTables.h`
- **Key feature**: Maintains both counts (for EM training) and normalized probabilities

#### Alignment Tables (ATables)
- **Purpose**: Store position-dependent alignment probabilities for Model 2
- **Representation**: a(j|i,l,m) - probability of target position j given source position i
- **Parameters**: Source position i, target position j, source length l, target length m
- **Location**: `GIZA++-v2/ATables.h`

#### Fertility Tables (NTables)
- **Purpose**: Model how many target words each source word generates
- **Representation**: n(φ|e) - probability that source word e has fertility φ
- **Range**: Fertility typically 0-10
- **Location**: `GIZA++-v2/NTables.h`

#### Distortion Tables (D3/D4/D5 Tables)
- **D3Tables**: Simple distortion for Model 3
- **D4Tables**: Class-based relative distortion for Model 4
  - Distinguishes head vs. non-head words in "cepts" (groups of target words from same source)
  - Uses word classes for generalization
- **D5Tables**: Vacancy-based distortion considering unfilled positions
- **Locations**: `GIZA++-v2/D4Tables.h`, `GIZA++-v2/D5Tables.h`

### Model Descriptions

#### IBM Model 1: Lexical Foundation

**Algorithm**:
```
Initialize: t(f|e) = uniform or from previous model

For each iteration:
  Clear count accumulators
  For each sentence pair (e, f):
    For each target word f_j:
      total = Σ_i t(f_j | e_i)  // Sum over all source words including NULL
      For each source word e_i:
        posterior = t(f_j | e_i) / total
        count[e_i, f_j] += posterior

  Normalize: t(f|e) = count(f|e) / Σ_f' count(f'|e)
```

**Key Properties**:
- Uniform alignment probability - no positional information
- Exact EM inference in polynomial time
- Viterbi alignment: For each target word, select source word with maximum t(f|e)
- Serves as initialization for more complex models

**Implementation**: `GIZA++-v2/model1.cpp`

#### IBM Model 2: Adding Position

**Probability Model**:
```
P(f, a | e) = ε(m|l) × ∏_j t(f_j | e_{a_j}) × a(a_j | j, l, m)
```

**Algorithm**:
```
For each iteration:
  Clear count accumulators
  For each sentence pair (e, f) with lengths l, m:
    For each target word f_j at position j:
      total = Σ_i t(f_j | e_i) × a(i | j, l, m)
      For each source word e_i at position i:
        posterior = t(f_j | e_i) × a(i | j, l, m) / total
        count_t[e_i, f_j] += posterior
        count_a[i, j, l, m] += posterior

  Normalize both t and a tables
```

**Key Properties**:
- Adds position-dependent alignment probability a(i|j,l,m)
- Still allows exact EM inference
- Alignment depends on absolute positions and sentence lengths
- Viterbi: argmax_i [t(f_j|e_i) × a(i|j,l,m)]

**Implementation**: `GIZA++-v2/model2.cpp`

#### HMM Alignment Model: First-Order Dependencies

**Probability Model**:
- Alignment positions form a first-order Markov chain
- Emission probabilities: t(f_j | e_i)
- Transition probabilities: p(a_j = i | a_{j-1} = i', word_classes)

**Algorithm** (Forward-Backward):
```
Build HMM with 2l states (l regular positions + l empty word positions)

Forward pass:
  α[i, j] = P(f_1...f_j, a_j = i | e)
  α[i, j] = t(f_j | e_i) × Σ_{i'} α[i', j-1] × p(i | i', classes)

Backward pass:
  β[i, j] = P(f_{j+1}...f_m | a_j = i, e)
  β[i, j] = Σ_{i'} t(f_{j+1} | e_{i'}) × p(i' | i, classes) × β[i', j+1]

Compute posteriors:
  γ[i, j] = P(a_j = i | e, f) = α[i, j] × β[i, j] / P(f | e)
  ε[i, i', j] = P(a_j = i, a_{j-1} = i' | e, f)

Accumulate fractional counts using γ and ε

Viterbi:
  Find best path using dynamic programming
```

**Key Properties**:
- Captures local alignment patterns better than Model 2
- Jump probability depends on distance between positions
- Uses word classes to generalize transition probabilities
- Exact inference via dynamic programming O(l² × m)
- Can model NULL alignments with empty word states

**Implementation**: `GIZA++-v2/hmm.cpp`, `GIZA++-v2/ForwardBackward.cpp`

#### IBM Model 3: Fertility-Based Alignment

**Probability Model**:
```
P(f, a | e) = p(m | l)
            × ∏_i n(φ_i | e_i)                    [fertility]
            × ∏_j t(f_j | e_{a_j})                [translation]
            × ∏_j d(j | a_j, l, m)                [distortion]
            × p0^{φ_0} × (1-p0)^{m-2φ_0}         [null insertion]
            × C(m, φ_0, φ_1, ..., φ_l)            [combinatorial]
```

Where:
- **φ_i**: Fertility of source word i (number of target words it generates)
- **φ_0**: Fertility of NULL (spurious insertions)
- **m_0 = φ_0**: Number of NULL-generated words
- **n(φ|e)**: Probability that word e has fertility φ
- **d(j|i,l,m)**: Distortion probability for position j given source position i

**Training Challenge**: Exponentially many possible alignments → exact EM is intractable!

**Solution**: Viterbi Training with Hill Climbing

```
For each iteration:
  Clear count accumulators
  For each sentence pair (e, f):
    1. Find initial alignment using Model 2 or HMM Viterbi
    2. Hill-climb to local optimum:
       repeat:
         changed = false
         for each possible MOVE or SWAP operation:
           if operation improves P(f, a | e):
             apply operation
             changed = true
       until no improvement
    3. Collect counts from this single best alignment

  Normalize all parameter tables (t, n, d, p0)
```

**Hill Climbing Operations**:
- **MOVE(i, j)**: Change alignment of target word j to source word i
- **SWAP(j₁, j₂)**: Swap alignments of two target words

**Optimization - MoveSwapMatrix**:
- Caches probability scores for potential moves and swaps
- Incrementally updated after each operation
- Avoids recomputing full alignment probability from scratch

**Key Properties**:
- Models fertility: one source word can generate multiple target words
- Fertility 0 models deletion; NULL fertility models insertion
- Distortion model accounts for word order differences
- Uses approximate inference (hill climbing) instead of exact EM
- More accurate than Model 2/HMM but computationally intensive

**Implementation**: `GIZA++-v2/model3.cpp`, `GIZA++-v2/model3_viterbi.cpp`, `GIZA++-v2/model3_viterbi_with_tricks.cpp`

#### IBM Model 4: Class-Based Relative Distortion

**Key Innovation**: Word classes and relative positioning

**Distortion Model**:
Model 4 distinguishes between:
- **Head words**: First target word generated by a source word
  - d₁(j - center(previous_cept) | word_classes)
  - Position relative to center of previous cept (group)
- **Non-head words**: Subsequent words from same source
  - d>₁(j - j_prev | word_classes)
  - Position relative to previous word in same cept

**Cept**: Set of target words aligned to the same source word

**Word Classes**:
- Words grouped into classes (typically 50-100 classes learned by mkcls)
- Distortion parameters depend on word classes rather than individual words
- Reduces parameter space and improves generalization

**Probability Model**:
Similar to Model 3, but with class-based relative distortion replacing d(j|i,l,m)

**Key Properties**:
- More linguistically motivated than Model 3's absolute distortion
- Captures phrasal cohesion (words in a cept tend to be adjacent)
- Word classes enable generalization across similar words
- Still uses Viterbi training with hill climbing

**Implementation**: `GIZA++-v2/transpair_model4.cpp`, `GIZA++-v2/D4Tables.h`

#### IBM Model 5: Vacancy-Based Distortion

**Key Innovation**: Models which target positions are "vacant" (unfilled)

**Distortion Model**:
- Tracks available positions in the target sentence
- Probability depends on number and distribution of vacancies
- More sophisticated than distance-based models

**Linguistic Motivation**:
- Words tend to fill gaps in the translation
- Position probability changes as sentence gets filled in
- Accounts for target sentence structure more explicitly

**Key Properties**:
- Most complex IBM model
- Often provides only marginal improvements over Model 4
- Computationally expensive
- Rarely used in default training (0 iterations typical)

**Implementation**: `GIZA++-v2/transpair_model5.cpp`, `GIZA++-v2/D5Tables.h`

### Viterbi Alignment Computation

**For Models 1-2 and HMM**:
- Exact dynamic programming in polynomial time
- Model 1: O(l × m) - argmax for each target word
- Model 2: O(l × m) - argmax for each target word
- HMM: O(l² × m) - dynamic programming over alignment paths

**For Models 3-5**:
1. **Initialize**: Use Model 2 or HMM Viterbi alignment as starting point
2. **Hill Climb**: Greedy local search using SWAP and MOVE operations
3. **Pegging** (optional): Fix high-confidence alignments to reduce search space
   - If P(a_j = i | e, f) > threshold, fix alignment j→i
   - Only search over unfixed positions
4. **Multiple Restarts**: Try different initializations for robustness

### EM Training Flow

**E-Step**:
- **Models 1-2, HMM**: Compute exact posterior probabilities over all alignments
- **Models 3-5**: Use single Viterbi alignment (Viterbi training approximation)

**M-Step**: All models normalize accumulated counts
```
θ_new(x|y) = count(x, y) / Σ_{x'} count(x', y)
```

**Parameter Transfer**:
```
Model 1 → Model 2: Initialize t from Model 1, random initialize a
Model 2 → HMM:     Initialize t from Model 2, random initialize transitions
HMM → Model 3:     Initialize t from HMM, estimate n and d from Viterbi alignments
Model 3 → Model 4: Keep t, n, p0; initialize d4 from d3
Model 4 → Model 5: Keep t, n, p0; initialize d5 from d4
```

### Performance Optimizations

The implementation includes numerous optimizations for efficiency:

1. **Sparse representation**: Only store non-zero probabilities
2. **Binary search for TTables**: Optional memory-efficient storage
3. **Pointer caching**: Reduce dereferencing in inner loops
4. **MoveSwapMatrix**: Cache for Model 3-5 hill climbing
5. **Incremental updates**: Update probabilities rather than recompute
6. **Word class abstraction**: Reduce parameter space in Model 4-5
7. **Pegging**: Fix high-confidence alignments to prune search space

### Output Files

GIZA++ produces rich output for analysis:

- **Translation tables** (*.t3.final, etc.): P(target|source) probabilities
- **Fertility tables** (*.n3.final): P(φ|source_word)
- **Alignment files** (*.A3.final): Viterbi alignments for each sentence pair
- **Perplexity files** (*.perp): Training and test perplexity by iteration
- **Configuration** (*.gizacfg): All parameters used for reproducibility

**Alignment File Format**:
```
# Sentence pair (1) source length 10 target length 12 alignment score 3.52e-19
il s' agit de la même société qui a changé de propriétaires
NULL ({ }) this ({ 1 2 }) is ({ 3 }) the ({ 4 }) same ({ 5 6 }) company ({ 7 }) that ({ 8 }) has ({ 9 }) changed ({ 10 11 }) owners ({ 12 })
```

Each source word is followed by the set of target positions it aligns to.

---

## mkcls Word Clustering Algorithm

### Purpose and Background

mkcls clusters words into classes using a maximum-likelihood criterion. Word classes are used in:

- **Language models**: Class-based n-gram models for smoothing
- **Translation models**: Reduce parameter space in alignment models (e.g., IBM Model 4)
- **Generalization**: Group similar words to combat data sparsity

The clustering optimizes the likelihood of a bigram class model, finding word-to-class assignments that best predict word co-occurrence patterns.

### Problem Formulation

**Input**:
- Vocabulary of N words
- Bigram statistics: counts of word pairs (w₁, w₂) in training corpus
- Number of classes K

**Output**:
- Mapping from each word to one of K classes

**Objective**: Find class assignment that maximizes the likelihood of the bigram class model

### Objective Functions

#### Maximum Likelihood (ML) - Default
```
Minimize: L(C) = -∑_{c1,c2} n(c₁,c₂) log P(c₂|c₁)
               = -∑_{c1,c2} n(c₁,c₂) log[n(c₁,c₂)/n₁(c₁)]
               = -∑_{c1,c2} n(c₁,c₂) log n(c₁,c₂) + ∑_{c1} n₁(c₁) log n₁(c₁)
```

Where:
- **n(c₁,c₂)**: Count of class bigram (c₁, c₂)
  - Sum of all word bigram counts where w₁ ∈ c₁ and w₂ ∈ c₂
- **n₁(c)**: Count of c as first element in bigrams
- **n₂(c)**: Count of c as second element in bigrams

This objective minimizes the perplexity of the class bigram model on the training data.

#### Leave-One-Out (LO)
```
Minimize: -∑_{c1,c2} n(c₁,c₂) log[n(c₁,c₂) - 1 - ρ]
          + ∑_c n₁(c) log[n₁(c) - 1]
          + ∑_c n₂(c) log[n₂(c) - 1]
          - η-correction
```

Includes leave-one-out smoothing with parameter ρ (default 0.75) to better estimate generalization.

#### Custom (MY)
Uses distortion function based on error function for smoothing, particularly for handling sparse bigram counts.

**Implementation**: `mkcls-v2/KategProblem.cpp:355-405`

### Core Data Structures

#### KategProblem Class

**Word-to-Category Mapping**:
- `_katOfWord[w]`: Current class assignment for word w

**Word Bigram Statistics** (KategProblemWBC):
- `wordFreq.after[w]`: List of words following w with counts
- `wordFreq.before[w]`: List of words preceding w with counts
- Sparse representation: only non-zero bigrams stored

**Category Bigram Statistics** (KategProblemKBC):
- `katFreq._n[c₁][c₂]`: Count of category bigram (c₁, c₂)
- `katFreq._n1[c]`: Total count of c as first in bigrams
- `katFreq._n2[c]`: Total count of c as second in bigrams

**Incremental Update Cache (NWG)**:
- Stores category distribution around a word
- Enables efficient computation of objective change when moving a word
- Avoids full recomputation from word bigrams

**Implementation**: `mkcls-v2/KategProblem.h`, `mkcls-v2/KategProblemWBC.h`, `mkcls-v2/KategProblemKBC.h`

### Optimization Algorithms

All optimization methods inherit from `IterOptimization` which provides:
- Main iteration loop
- Best solution tracking
- Termination criteria
- Statistics collection

Methods differ in their **acceptance criteria** for candidate moves.

#### Hill Climbing (HC)

**Accept Rule**: `delta < 0` (only accept improvements)

**Algorithm**:
```
current = initial_clustering
best = current
while not converged:
  candidate_move = select_word_and_target_class()
  delta = change_in_objective(candidate_move)
  if delta < 0:
    apply(candidate_move)
    current_value += delta
    if current_value < best_value:
      best = current
```

**Properties**:
- Simplest method
- Fast but gets stuck in local minima
- Deterministic given same initialization

**Implementation**: `mkcls-v2/HCOptimization.cpp`

#### Greedy Descent Algorithm (GDA)

**Accept Rule**: `current_value + delta < temperature`

**Temperature Update**: Adaptive cooling based on current objective
```
T_new = T - α × (T - current_value)
```
where α = 0.001 (default)

**Properties**:
- Maintains a threshold that adapts to current objective value
- More flexible than pure hill climbing
- Can escape some local minima early in search

**Implementation**: `mkcls-v2/GDAOptimization.cpp`

#### Threshold Accepting (TA)

**Accept Rule**: `delta < temperature`

**Temperature Update**: Linear decrease
```
T_k = T_0 - k × δT
```

**Initialization**:
1. Run sampling phase to estimate delta distribution
2. Set T_0 such that `defaultAnnRate` (40%) of deltas would be accepted
3. Compute δT to reach T_final ≈ 0 after specified iterations

**Properties**:
- Simpler than Simulated Annealing (no probabilistic acceptance)
- Accepts worsening moves up to threshold
- Good balance of simplicity and effectiveness
- **Default algorithm** in mkcls

**Implementation**: `mkcls-v2/TAOptimization.cpp`

#### Simulated Annealing (SA)

**Accept Rule**:
```
if delta < 0:
  accept
else:
  accept with probability exp(-delta / T)
```

**Temperature Update**: Geometric cooling
```
Every 'schrittzahl' steps: T = T × α
```
where α = 0.95 (default)

**Initialization**:
1. Sample to find initial temperature where `defaultAnfAnnRate` (90%) of worsening moves accepted
2. Compute cooling schedule to reach `defaultEndAnnRate` (10⁻⁹) by end

**Properties**:
- Classic probabilistic optimization method
- Theoretical convergence guarantees (with appropriate cooling)
- Most sophisticated algorithm
- Computationally expensive due to random number generation

**Implementation**: `mkcls-v2/SAOptimization.cpp`

#### Record-to-Record Travel (RRT)

**Accept Rule**: Based on deviation from best solution found
```
accept if: current_value + delta < best_value + threshold
```

**Properties**:
- Similar to Threshold Accepting
- Tracks best solution and accepts moves within threshold of it
- Can be more robust than TA

**Implementation**: `mkcls-v2/RRTOptimization.cpp`

#### Moving-Segment-Best (MSB)

**Properties**:
- Evaluates moving contiguous segments of words
- More sophisticated neighborhood structure

**Implementation**: `mkcls-v2/MSBOptimization.cpp`

### Move Generation Strategies

The algorithm uses local search where each move transfers one word to a different class.

#### Word Selection (`wortwahl`)

**W_RAN (Random)**:
- Select word uniformly at random
- Gives all words equal probability of moving

**W_DET_DECR (Deterministic Decreasing)** - Default:
- Cycle through words in decreasing frequency order
- High-frequency words moved more often
- Rationale: Frequent words have more impact on objective

**W_DET_INCR (Deterministic Increasing)**:
- Cycle through words in increasing frequency order
- Rare words moved more often

#### Category Selection (`katwahl`)

**K_RAN (Random)**:
- Select target category uniformly at random
- Fast but may waste evaluations on poor moves

**K_DET (Deterministic)**:
- Rotate through all K categories systematically
- Ensures all options explored

**K_BEST (Best)** - Default for TA/SA:
- Evaluate moving the word to ALL categories
- Select the category yielding minimum objective increase
- Most expensive but highest quality moves
- Amortized cost acceptable since good moves are rare

**Implementation**: `mkcls-v2/KategProblem.cpp:459` (`_change` method)

### Incremental Objective Computation

The key to efficiency is computing the change in objective when moving a word without recomputing the full objective.

**Algorithm** (`_valueChange` method):

```
Given: move word w from class c_old to class c_new

1. Build NWG (Neighbor Word-to-Category) structure for w:
   - For each word w' with bigram (w, w') or (w', w):
     - Record class of w' and bigram count
   - This gives category context around w

2. Compute delta for affected category bigrams:
   For each category c in NWG[w]:
     old_value += h(n(c_old, c)) + h(n(c, c_old))
     new_value += h(n(c_old, c) - Δ) + h(n(c, c_old) - Δ)
     new_value += h(n(c_new, c) + Δ) + h(n(c, c_new) + Δ)

   where Δ = count of (w, w') or (w', w) bigrams for w' in class c

3. Compute delta for unigram terms:
   old_value += h(n1(c_old)) + h(n2(c_old)) + h(n1(c_new)) + h(n2(c_new))
   new_value += h(n1(c_old) - w.n1) + h(n2(c_old) - w.n2)
              + h(n1(c_new) + w.n1) + h(n2(c_new) + w.n2)

4. Return: delta = new_value - old_value
```

Where `h(x) = x log x` for ML criterion (precomputed for small x).

**Efficiency**:
- Only evaluates O(K) category bigrams (where K = number of categories)
- Independent of total vocabulary size
- NWG caching amortizes cost across multiple category evaluations

**Implementation**: `mkcls-v2/KategProblem.cpp:666` (`_valueChange` method)

### Initialization Strategies

#### INIT_RAN (Random)
- Assign each word to random category
- Fast but poor starting point

#### INIT_AIO (All-In-One)
- All words start in single category
- Other categories empty initially

#### INIT_FREQ (Frequency-Based)
- Sort words by frequency
- Assign to categories in round-robin fashion
- Groups similarly-frequent words

#### INIT_LWRW (Local Word-to-Word)** - Often Default:
- 50% most frequent words each get their own class
- Remaining words grouped based on co-occurrence with frequent words
- Provides linguistically motivated initial clustering

#### INIT_OTHER (From File)
- Load initial clustering from file
- Enables warm-starting or refining existing clusters

**Implementation**: `mkcls-v2/KategProblem.cpp:144` (`_initialize` method)

### Complete Algorithm Flow

**Main Process** (`mkcls.cpp:makeIterOpt` → `solveProblem`):

```
1. Parse command-line arguments:
   - Input file (word bigram statistics)
   - Number of classes K
   - Optimization method (TA/SA/HC/GDA/RRT/MSB)
   - Number of iterations
   - Number of random restarts

2. Load word bigram statistics from corpus

3. For each random restart:
   a. Create KategProblem with bigram statistics

   b. Initialize clustering (INIT strategy)

   c. Create optimizer (TA/SA/etc.) with KategProblem

   d. Run minimize():
      step = 0
      while step < max_steps and not_converged:
        - Generate candidate move (change())
          - Select word (W_RAN/W_DET_DECR/W_DET_INCR)
          - Select target category (K_RAN/K_DET/K_BEST)

        - Compute objective change (valueChange())
          - Incremental computation using NWG

        - Update temperature/threshold (abkuehlen())
          - Method-specific cooling schedule

        - Decide acceptance (accept(delta))
          - HC: delta < 0
          - TA: delta < T
          - SA: delta < 0 or random() < exp(-delta/T)
          - etc.

        - If accepted:
          - Apply change (doChange())
          - Update category bigram statistics
          - Update current objective value

        - Track best solution seen so far

        - Check convergence:
          - Maximum steps reached
          - No improvement for maxNonBetterIterations

        step++

   e. Record best clustering from this restart

4. Select overall best clustering across all restarts

5. Output:
   - Write class assignments to file
   - Print statistics (objective value, iterations, etc.)
```

### Performance Optimizations

1. **Sparse Storage**: Only non-zero word bigrams stored in memory

2. **Incremental Updates**: Category bigram counts updated incrementally when words move

3. **Precomputed Tables**:
   - log(x) and x×log(x) cached for small integers (x < 4000)
   - Avoids repeated expensive math operations

4. **NWG Caching**:
   - Category context around word computed once per word selection
   - Reused across all K category evaluations in K_BEST

5. **Lazy Evaluation**: NWG only recomputed when word selection changes

6. **Efficient Data Structures**:
   - Hash maps for sparse bigram storage
   - Arrays for dense category statistics
   - Optimized for cache locality in inner loops

### Typical Usage

**Command Line**:
```bash
mkcls -c50 -n10 -ptrainfile -Voutput.classes
```

**Parameters**:
- `-c50`: Create 50 classes
- `-n10`: Run 10 optimization iterations
- `-p`: Input file (word bigram statistics or text)
- `-V`: Output file (word-to-class mapping)

**Additional Options**:
- `-iINIT_TYPE`: Initialization strategy (ran/aio/freq/lwrw)
- `-oOPT_METHOD`: Optimization method (ta/sa/hc/gda/rrt/msb)
- `-rN`: Number of random restarts (default: 1)

### Output Format

**Class File** (one word per line):
```
word1  class_id1
word2  class_id2
word3  class_id3
...
```

Class IDs are integers from 0 to K-1.

### Integration with GIZA++

In GIZA++ training pipeline:

1. **Train mkcls** on source and target vocabularies:
   ```bash
   mkcls -c50 -n10 -psource.txt -Vsource.classes
   mkcls -c50 -n10 -ptarget.txt -Vtarget.classes
   ```

2. **Use in GIZA++**: Provide class files via `-SourceVocabularyclasses` and `-TargetVocabularyclasses`

3. **Model 4 Training**: Uses word classes in distortion model d₁ and d>₁

4. **Effect**: Reduces parameter space, improves generalization, better alignments for rare words

---

## References

### GIZA++ Papers

1. **Och & Ney (2000a)**: "Improved Statistical Alignment Models"
   - Proceedings of ACL 2000, pp. 440-447
   - Describes improvements in GIZA++ over original GIZA

2. **Och & Ney (2000b)**: "A Comparison of Alignment Models for Statistical Machine Translation"
   - Proceedings of COLING 2000, pp. 1086-1090
   - Empirical comparison of IBM Models and HMM

3. **Brown et al. (1993)**: "The Mathematics of Statistical Machine Translation: Parameter Estimation"
   - Computational Linguistics, 19(2):263-311
   - Original description of IBM Models 1-5

4. **Vogel et al. (1996)**: "HMM-Based Word Alignment in Statistical Translation"
   - Proceedings of COLING 1996, pp. 836-841
   - HMM alignment model

### mkcls Papers

1. **Och (1999)**: "An Efficient Method for Determining Bilingual Word Classes"
   - Proceedings of EACL 1999
   - Describes the mkcls algorithm and optimization methods

### Additional Resources

- GIZA++ Homepage: http://www.fjoch.com/GIZA++.html
- mkcls Homepage: http://www.fjoch.com/mkcls.html
- Moses SMT Toolkit: http://www.statmt.org/moses/
- mgiza (Multi-threaded GIZA++): https://github.com/moses-smt/mgiza

---

## Summary

**GIZA++** and **mkcls** represent foundational tools in statistical machine translation:

- **GIZA++** implements a sophisticated cascade of alignment models, from simple lexical models (IBM-1) to complex fertility and distortion models (IBM 3-5), using a combination of exact EM inference (Models 1-2, HMM) and Viterbi approximation with hill-climbing (Models 3-5).

- **mkcls** solves a combinatorial optimization problem to cluster words into classes, using local search algorithms (HC, TA, SA) with efficient incremental objective computation to maximize bigram class model likelihood.

Both tools exemplify the practical application of statistical modeling and optimization in NLP, with careful attention to algorithmic efficiency and engineering trade-offs between computational cost and model quality.
