---
title: "Building a Production-Ready Semantic Chunker: Valley Detection, Go, and pgvector"
date: 2026-07-15
authors: amit
---

> *Most RAG tutorials hand you a chunker. `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, a LangChain convenience wrapper. You set a size, maybe an overlap, and move on. This post is about what happens when that isn't enough, and how to build something better in ~150 lines of pure, dependency-free Go.*

---

## Table of Contents

1. [The Problem: Why Fixed-Size Chunking Breaks](#1-the-problem)
2. [Architecture](#2-architecture)
3. [The Algorithm: Four Steps to Semantic Cuts](#3-the-algorithm)
   - [Step 1: Embed Every Atom](#step-1--embed-every-atom)
   - [Step 2: Smooth with Neighbors](#step-2--smooth-with-neighbors)
   - [Step 3: Compute the Distance Curve](#step-3--compute-the-distance-curve)
   - [Step 4: Cut at the Valleys](#step-4--cut-at-the-valleys)
4. [Size Guards: Safety Rails That Respect Semantics](#4-size-guards)
5. [The Infra Layer: pgvector, halfvec, and Idempotent Stages](#5-the-infra-layer)
6. [The Go + Python Boundary](#6-the-go--python-boundary)
7. [Testing Pure Logic Without Mocks](#7-testing-pure-logic-without-mocks)
8. [Summary](#8-summary)

---

## 1. The Problem

Consider any long-form audio recording like a podcast, a lecture series, or an interview archive. In multilingual regions, it's common for a speaker to switch freely between two languages, sometimes mid-sentence. A bilingual lecture transcript might look like this:

```
"Iska matlab hai, the observer and the observed are not two things."
"Jab tum sach mein dekh rahe ho, toh observer hi observation ban jata hai."
"This is what most mindfulness teachers miss when they talk about awareness."
"Aur yahi baat hai jo Western psychology abhi begin kar rahi hai samajhna."
```

Now imagine splitting that with `chunk_size=512, overlap=50`. You will get cuts in the middle of thoughts, leaving one half of a complete argument on each side of a boundary. Instead of coherent ideas, you get syntactically broken, incomplete fragments that degrade embedding quality and confuse the retriever.

The core problem with fixed-size chunking is that **it is blind to meaning**. It knows characters and tokens. It does not know when the speaker finishes one idea and starts another.

What we actually want:

```mermaid
graph LR
    A["Topic: Observer & awareness\n(atoms 0–4)"] -->|semantic break| B["Topic: Psychology & perception\n(atoms 5–9)"]
    B -->|semantic break| C["Topic: Practice & technique\n(atoms 10–14)"]
```

Chunks that track *ideas*, not byte offsets.

> **What is an "atom"?**
> An atom is our smallest indivisible unit of text—typically a single complete sentence or a discrete speech utterance. By operating on whole atoms, we ensure a sentence is never chopped in half.
> 
> For example, in the bilingual transcript above:
> - **Atom 0:** `"Iska matlab hai, the observer and the observed are not two things."`
> - **Atom 1:** `"Jab tum sach mein dekh rahe ho, toh observer hi observation ban jata hai."`
> 
> No matter the chunk size limits, these atoms remain fully intact. They may be grouped together or separated at a semantic boundary, but they will never be split down the middle.

---


## 2. Architecture

To keep the core chunking algorithm isolated and testable without a database or GPU, you can use a strict four-layer architecture for your components:

```mermaid
graph TD
    T["transport.go\n(cobra CLI flags)"]
    H["handler.go\n(orchestration, idempotency check)"]
    U["usecase.go\n(pure business logic, chunkBlock)"]
    R["repository/\n(store, mlclient)"]

    T --> H --> U --> R

    style U fill:#1e3a5f,color:#fff
    style R fill:#1a2e1a,color:#fff
```

By keeping the pure business logic in the `usecase` layer, the semantic chunking algorithm can be tested independently of Postgres or the embedding model.

---

## 3. The Algorithm: Four Steps to Semantic Cuts

Here is the full chunking flow for one text block:

```mermaid
flowchart TD
    A["Text block\n(a collection of atoms 0..N)"]
    B["Embed each atom\nBGE-M3 → 1024-d vectors"]
    C["Smooth with ±1 neighbors\nre-normalize"]
    D["Compute consecutive\ncosine distances"]
    E["Detect valley cuts\n@ p92 percentile threshold"]
    F["Build spans\nfrom cut indices"]
    G["Apply size guards\nmerge undersize → split oversize"]
    H["Emit Chunks\nwith seq index + core text"]

    A --> B --> C --> D --> E --> F --> G --> H
```

### Step 1: Embed Every Atom

Each **atom** (a single sentence or distinct utterance) is embedded via **BGE-M3**, a multilingual 1024-dimensional dense encoder that handles Hindi, English, and Hinglish in the same vector space.

```go
// usecase.go
vecs, err := u.embed.Embed(ctx, texts, true) // normalize=true → unit vectors
```

BGE-M3 is called via a **Python FastAPI sidecar** (the `pyworker`). Go owns all orchestration and logic; Python is confined strictly to GPU-bound inference. The HTTP contract is minimal: a list of strings in, a list of float32 slices out.

```mermaid
sequenceDiagram
    participant Go as Go usecase
    participant Worker as pyworker /embed
    participant BGE as BGE-M3 (GPU)

    Go->>Worker: POST /embed {texts: [...N atoms...], normalize: true}
    Worker->>BGE: FlagModel.encode(atoms, batch_size=16)
    BGE-->>Worker: float32[N][1024] dense vectors
    Worker-->>Go: {embeddings: [[...], ...]}
```

The sidecar uses **double-checked locking** to load BGE-M3 once and reuse across requests:

```python
# models.py
def get_bge():
    global _bge_model
    if _bge_model is None:
        with _lock:
            if _bge_model is None:
                from FlagEmbedding import BGEM3FlagModel
                _bge_model = BGEM3FlagModel(settings.bge_model, use_fp16=settings.use_fp16)
    return _bge_model
```

---

### Step 2: Smooth with Neighbors

Raw per-sentence embeddings are noisy. A single noisy or off-topic sentence can spike a distance value falsely. 

> **Why smooth?** 
> When people speak naturally, they often throw in brief, off-topic fillers (e.g., *"Hold on, let me take a sip of water"*). If we don't smooth the vectors, this single sentence creates a massive distance "spike," tricking the algorithm into cutting the chunk in half. By averaging it with its neighbors, we pull that outlier back toward the main topic and prevent a false cut.

We reduce this noise by replacing each atom's vector with the **mean of itself and its ±1 neighbors**, then re-normalizing to unit length:

```go
// boundary.go
func SmoothNeighbors(vecs [][]float32) [][]float32 {
    // for each atom i: average vecs[i-1], vecs[i], vecs[i+1]
    // then L2-normalize the mean
}
```

```mermaid
graph LR
    v0["v[0]"] --> sm0["smooth[0]\n= norm(v[0]+v[1])"]
    v1["v[1]"] --> sm1["smooth[1]\n= norm(v[0]+v[1]+v[2])"]
    v2["v[2]"] --> sm2["smooth[2]\n= norm(v[1]+v[2]+v[3])"]
    v3["v[3]"] --> sm3["smooth[3]\n= norm(v[2]+v[3])"]

    style sm1 fill:#1e3a5f,color:#fff
    style sm2 fill:#1e3a5f,color:#fff
```

Edge atoms (first and last) only have two neighbors to average. Boundary handling is automatic because we skip out-of-range indices.

---

### Step 3: Compute the Distance Curve

With smoothed, unit-norm vectors, we compute cosine distance between each pair of **consecutive** atoms:

```
distance[i] = 1 - cosine_similarity(smooth[i], smooth[i+1])
```

For unit vectors, cosine distance is simply `1 - dot(a, b)`, which is cheap and numerically clean.

```mermaid
xychart-beta
    title "Cosine distance curve (example: 10-atom block)"
    x-axis ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"]
    y-axis "distance" 0 --> 1
    bar [0.08, 0.11, 0.07, 0.79, 0.09, 0.12, 0.06, 0.83, 0.10]
```

The tall bars at positions `3→4` and `7→8` are where the topic shifts. Everything else is intra-topic variation.

---

### Step 4: Cut at the Valleys

Instead of a fixed absolute threshold, we use a **per-file percentile**:

```go
// boundary.go
func DetectValleyCuts(distances []float64, percentile float64) []int {
    threshold := Percentile(distances, percentile) // e.g. p92
    var cuts []int
    for i, d := range distances {
        if d >= threshold {
            cuts = append(cuts, i)
        }
    }
    return cuts
}
```

Why percentile instead of absolute threshold?

```mermaid
graph LR
    subgraph "Absolute threshold = 0.5"
        A1["Short document\n(max distance = 0.4)"] -->|"0 cuts"| A2["❌ Whole file one chunk"]
        A3["Noisy document\n(many small spikes)"] -->|"too many cuts"| A4["❌ Tiny fragments"]
    end

    subgraph "Percentile threshold = p92"
        B1["Short document"] -->|"top 8% distances"| B2["✅ Adapts to file"]
        B3["Noisy document"] -->|"top 8% distances"| B4["✅ Adapts to file"]
    end
```

With `valley_percentile: 0.92` (from `config.yaml`), exactly the top 8% of cosine-distance values in a given block become cut points. A shorter, lower-variance block naturally gets fewer cuts; a longer, denser block gets more cuts automatically, without any per-file tuning.

---

## 4. Size Guards

Valley detection proposes cuts. Size guards **correct** them. The two-pass precedence rule is: **merge undersize first, then split oversize**. This ordering matters because you never want to split something you are about to merge.

```mermaid
flowchart TD
    V["Valley cuts propose N spans"]
    M["Pass 1: Merge undersize\nspanWords < minWords?"]
    S["Pass 2: Split oversize\nspanWords > maxWords?"]
    Out["Final spans\nminWords <= each <= maxWords\n(single-atom exception)"]

    V --> M --> S --> Out
```

### Merge: pick the more-similar neighbor

An undersize span always merges toward the neighbor it is **semantically closer to**, meaning the side with the smaller cosine distance at the seam:

```mermaid
graph LR
    A["Span A\n100 words"] -->|"seam dist = 0.1\n(similar)"| B["Span B\n10 words - undersize"]
    B -->|"seam dist = 0.9\n(different)"| C["Span C\n100 words"]
    B -->|"merges LEFT\n(more similar)"| A

    style B fill:#5c1a1a,color:#fff
```

This is semantics-aware merging. Even when forced to merge, the algorithm respects the content.

### Split: find the deepest internal valley

When a span is too large, we split at the **strongest internal valley** within a sensible size band. If no valley exists in the band, we fall back to the target word count:

```mermaid
graph TD
    Oversize["Oversize span: atoms 0..5, 300 words, maxWords=120"]
    Valley["Strongest valley at atom 2-to-3, distance=0.9"]
    Split["Split: [0..2] + [3..5], 150w + 150w"]
    Again["Still oversize -> recurse"]
    Final["[0..1] + [2..2] + [3..4] + [5..5]"]

    Oversize --> Valley --> Split --> Again --> Final
```

The documented **whole-sentence exception**: a single atom (one sentence) that exceeds `maxWords` is returned as-is. It cannot be split further without breaking sentence integrity.

### Config snapshot

```yaml
# config.example.yaml
chunk_configs:
  default:
    target_words: 135
    min_words:    40
    max_words:    475
    valley_percentile: 0.92
  small:                       # for parameter sweeps
    target_words: 90
    min_words:    30
    max_words:    350
    valley_percentile: 0.90
```

Multiple `chunk_config` values coexist in the database under the same source text. You can run a parameter sweep (default vs. small) without re-embedding the source, since the per-sentence embeddings are stable; only the chunking groupings change.

---

## 5. The Infra Layer

### Idempotent stages

Each stage is independently re-runnable. The contract for the `chunk` stage: **delete existing chunks for this `(file_id, chunk_config)` pair, then regenerate**. Source text segments are never touched.

```mermaid
sequenceDiagram
    participant CLI as pipeline chunk
    participant H as handler
    participant Store as Postgres

    CLI->>H: Chunk(fileID, chunkConfig)
    H->>Store: DeleteChunksByConfig(fileID, chunkConfig)
    H->>Store: GetLangSegments(fileID)
    loop each text block
        H->>Store: GetSegmentsByLangSeg(langSegID)
        H->>H: chunkBlock(segments, config)
    end
    H->>Store: SaveChunks(all)
```

Re-running the same command is safe. Changing `chunk_config` parameters re-generates only the chunks, not the source segments.

### pgvector schema: halfvec for two models

The `chunks` table stores **two embedding columns** side by side:

```sql
-- migrations/0001_init.sql (simplified)
CREATE TABLE chunks (
    chunk_id       BIGSERIAL PRIMARY KEY,
    file_id        TEXT       NOT NULL REFERENCES files(file_id),
    lang_seg_id    BIGINT     NOT NULL REFERENCES language_segments(lang_seg_id),
    language       TEXT       NOT NULL,          -- 'hi' | 'en'
    seq_index      INT        NOT NULL,
    start_time     FLOAT8     NOT NULL,
    end_time       FLOAT8     NOT NULL,
    core_text      TEXT       NOT NULL,
    word_count     INT        NOT NULL,
    chunk_config   TEXT       NOT NULL,

    bge_vec        halfvec(1024),   -- BGE-M3 dense, L2-normalized
    gemini_vec     halfvec(3072)    -- gemini-embedding-001, L2-normalized
);

CREATE INDEX chunks_bge_hnsw    ON chunks USING hnsw (bge_vec    halfvec_ip_ops);
CREATE INDEX chunks_gemini_hnsw ON chunks USING hnsw (gemini_vec halfvec_ip_ops);
```

`halfvec` (16-bit float) instead of `vector` (32-bit) halves storage and index size with negligible precision loss for normalized retrieval. `halfvec_ip_ops` (inner product) is equivalent to cosine similarity when vectors are L2-normalized — which both BGE-M3 and Gemini vectors are enforced to be before write.

```mermaid
graph LR
    BGE["BGE-M3\n1024-d float32\nasserted unit-norm"] -->|"UPDATE chunks SET bge_vec"| PG["chunks table\nhalfvec(1024)\nHNSW index"]
    Gemini["gemini-embedding-001\n3072-d float32\nnormalized defensively"] -->|"UPDATE chunks SET gemini_vec"| PG2["chunks table\nhalfvec(3072)\nHNSW index"]
```

### Embedding provenance

Every embedding run writes a row to `embedding_runs` to let you trace which model version produced which vectors:

```mermaid
erDiagram
    embedding_runs {
        bigint embedding_run_id PK
        text   column_name "bge_vec or gemini_vec"
        text   model_name
        text   model_version
        int    dim
        bool   normalized
        text   chunk_config
        timestamptz created_at
    }
    chunks ||--o{ embedding_runs : "chunk_config matches"
```

---

## 6. The Go + Python Boundary

The architecture is deliberately asymmetric:

```mermaid
graph TD
    subgraph "Go — orchestrator"
        direction TB
        G1["Pipeline orchestration"]
        G2["Semantic chunking / guards / boundary"]
        G3["Postgres + pgvector"]
        G5["Embedding APIs"]
        G6["Eval harness"]
    end

    subgraph "Python — embedding sidecar"
        direction TB
        P3["BGE-M3\n(dense embedding)"]
    end

    G1 <-->|"HTTP localhost:8000"| P3

    style G1 fill:#1e3a5f,color:#fff
    style P3 fill:#1a2e1a,color:#fff
```

**Why this split?**

- Go is fast, statically typed, and deploys as a single binary
- Python is where the GPU model ecosystem lives; BGE-M3 requires PyTorch and FlagEmbedding
- The HTTP boundary is narrow and typed: a list of strings in, a list of float32 slices out
- **Stub mode** (`PYWORKER_STUB=1`): Python returns deterministic unit-norm vectors, so the full chunking pipeline runs end-to-end on a laptop with zero GPU

---

## 7. Testing Pure Logic Without Mocks

By designing your core algorithm (valley detection and size guards) as **pure logic** with no I/O or database dependencies, testing becomes incredibly easy. You don't need mocks, test databases, or running API workers. You just pass in arrays of numbers and assert the exact output.

For example, testing the valley detection logic is as simple as:

```go
// boundary_test.go
func TestDetectValleyCuts(t *testing.T) {
    // One clear spike at index 2.
    distances := []float64{0.1, 0.1, 0.9, 0.1, 0.1}
    cuts := DetectValleyCuts(distances, 0.9)
    if len(cuts) != 1 || cuts[0] != 2 {
        t.Fatalf("cuts=%v want [2]", cuts)
    }
}

// guards_test.go
func TestApplySizeGuards_MergeDirectionMoreSimilar(t *testing.T) {
    // Middle span undersize; left seam (0.1) more similar than right (0.9)
    // → must merge LEFT
    atomWords := []int{100, 10, 100}
    spans     := []Span{{0, 0}, {1, 1}, {2, 2}}
    distances := []float64{0.1, 0.9}
    out := ApplySizeGuards(spans, atomWords, distances,
        SizeGuards{MinWords: 40, TargetWords: 120, MaxWords: 500})
    // expect: [{0,1}, {2,2}]
}
```

When building your own system, you can easily create a comprehensive test suite to cover all edge cases without any complex setup:

```mermaid
mindmap
  root((Test coverage))
    Valley Detection
      L2 normalization
      Cosine distance: identical vectors
      Cosine distance: orthogonal vectors
      Percentile at edges
      Valley detection at known spike
      Smoothing re-normalization invariant
    Size Guards
      Span building from cuts
      Edge cuts ignored
      Merge undersize
      Merge toward more-similar neighbor
      Oversize split: contiguous cover
      Oversize split: no span exceeds max
      Single unsplittable atom exception
```

This ensures your chunking logic is perfectly reliable before it ever touches a real database or embedding model.

---

## 8. Summary

```mermaid
mindmap
  root((Semantic Chunking))
    Algorithm
      Embed atoms with BGE-M3
      Smooth with neighbors
      Cosine distance curve
      Percentile-threshold valleys
      Two-pass size guards
    Architecture
      Pure logic separated from I/O
      Idempotent pipeline stages
    Infra
      pgvector halfvec for two models
      HNSW on inner product ops
      Stub mode for local dev
    Testing
      Zero mocks for pure logic
      Table-driven unit tests
      go test with no services
```

1. **Percentile thresholding beats absolute thresholds.** Your threshold adapts to each file's variance distribution automatically without per-file tuning.

2. **Smooth before computing distances.** A single noisy sentence creates a false spike; ±1 neighbor averaging removes most of them cheaply.

3. **Merge before split in size guards.** Merging can create an oversize span — which is then handled. The reverse causes subtler bugs.

4. **Pure-logic separation is not just an architecture virtue.** Your most complex algorithm becomes testable in five lines, portable, and auditable without any framework dependency.

5. **`halfvec` in pgvector** is the right default for normalized vectors: halved storage, identical retrieval semantics (inner product equals cosine for unit vectors), HNSW indexable on vectors that would otherwise be too wide.

6. **Design stub mode in from day one.** A Python embedding sidecar with a stub fallback (`PYWORKER_STUB=1`) makes local development as fast as any pure-Go project. Do not bolt it on later.

