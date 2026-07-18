---
title: "halfvec, Matryoshka, or Nothing: Picking a Vector Storage Strategy"
description: "Three paths to storing vectors in a RAG system, and a decision framework for knowing which one will hurt you. Covers halfvec quantization, Matryoshka truncation, the pgvector HNSW 2000-dim limit, and how to measure before you commit."
date: 2026-07-10
authors: amit
---

> *Premature optimization in vector storage is a silent bug. It doesn't crash. It just quietly makes your AI dumber, and you won't notice until a user does.*

---

## Table of Contents

1. [The Problem: The Storage Dilemma](#1-the-problem-the-storage-dilemma)
2. [The Three Paths at a Glance](#2-the-three-paths-at-a-glance)
3. [Path 1: No Optimization](#3-path-1-no-optimization)
4. [Path 2: Matryoshka Truncation](#4-path-2-matryoshka-truncation)
5. [Path 3: halfvec Quantization](#5-path-3-halfvec-quantization)
6. [Side-by-Side Comparison](#6-side-by-side-comparison)
7. [The Decision Framework](#7-the-decision-framework)
8. [The Domain Nuance Problem](#8-the-domain-nuance-problem)
9. [What Silent Degradation Actually Looks Like](#9-what-silent-degradation-actually-looks-like)
10. [Prove It: The Gold Standard Eval Loop](#10-prove-it-the-gold-standard-eval-loop)

---

## 1. The Problem: The Storage Dilemma

Here's the thing about modern embedding models: they're heavy.

- `gemini-embedding-002` puts out **3,072 dimensions** per chunk
- `BGE-M3` puts out **1,024 dimensions** per chunk
- `text-embedding-3-large` is also **3,072 dimensions**

Each dimension is a 32-bit float, 4 bytes. Scale that to 10 million chunks and the numbers get uncomfortable fast:

```mermaid
graph TD
    A["10M chunks x 3072 dims x 4 bytes"]
    B["~120 GB of raw vector data"]
    C["Plus HNSW index overhead: 1.5-2x"]
    D["Realistic RAM requirement: 180-240 GB"]

    A --> B --> C --> D

    style D fill:#5c1a1a,stroke:#7c2929,color:#fff
    style B fill:#374151,stroke:#9CA3AF,color:#fff
```

At that scale, the instinct to optimize is fair. The danger is optimizing without knowing what you're actually giving up.

---

## 2. The Three Paths at a Glance

There are three choices. Exactly three. Every "optimization strategy" you'll read about is one of these.

```mermaid
flowchart TD
    Original["Original Vector\n3072 dims / 32-bit float\n~12 KB per vector"]

    Original -->|Path 1| P1["No Optimization\n3072 dims / 32-bit\nAll data preserved"]
    Original -->|Path 2| P2["Matryoshka Truncation\n768 dims / 32-bit\nReduce WIDTH"]
    Original -->|Path 3| P3["halfvec Quantization\n3072 dims / 16-bit\nReduce DEPTH"]

    P1 --> R1["Memory: 12 KB\nAccuracy: MAX\nCompatibility: Full"]
    P2 --> R2["Memory: 3 KB  (-75%)\nAccuracy: GOOD for broad queries\nCompatibility: MRL models only"]
    P3 --> R3["Memory: 6 KB  (-50%)\nAccuracy: HIGH for all query types\nCompatibility: Any model"]

    style P1 fill:#374151,stroke:#9CA3AF,color:#fff
    style P2 fill:#F59E0B,stroke:#B45309,color:#fff
    style P3 fill:#10B981,stroke:#047857,color:#fff
    style R1 fill:#374151,stroke:#9CA3AF,color:#fff
    style R2 fill:#92400e,stroke:#B45309,color:#fff
    style R3 fill:#065f46,stroke:#047857,color:#fff
```

Each one has a legitimate place. The mistake is picking one by default, without checking if the tradeoff actually fits your situation.

---

## 3. Path 1: No Optimization

> **Ask yourself first:** is my dataset actually big enough to justify this?

Choosing not to optimize is a real architectural decision. It's not laziness. Sometimes it's the right call.

### The math

```mermaid
xychart-beta
    title "Storage Required vs Corpus Size (1024-dim, 32-bit vectors)"
    x-axis ["10K chunks", "50K chunks", "100K chunks", "500K chunks", "1M chunks", "10M chunks"]
    y-axis "Storage (GB)" 0 --> 40
    bar [0.04, 0.2, 0.4, 2, 4, 40]
```

100,000 chunks, roughly a corpus of 500 books, costs **400 MB** unoptimized. That fits in a `db.t3.medium`. You don't need to do anything.

The point where "no optimization" starts to genuinely hurt is around **5M+ chunks** for 1024-dim models, or **2M+ chunks** for 3072-dim models. Below that, you're probably optimizing for a problem you don't have yet.

### When to use it

- Under 1M chunks
- No hard RAM budget
- Retrieval quality is what you're measured on
- You want nothing extra to debug

### My rule

> If you don't have a real constraint, don't manufacture one. Every optimization layer is a debugging surface you'll have to explain at 2am when retrieval breaks.

---

## 4. Path 2: Matryoshka Truncation

Matryoshka Representation Learning (MRL) is a training technique where the model packs its most important, broadest semantic concepts into the **earliest dimensions**, and pushes fine-grained nuance toward the **later dimensions**.

Think of it like a Russian nesting doll. The first doll gives you the rough shape. The inner dolls add detail.

```mermaid
flowchart LR
    subgraph early["Dims 0-256: Coarse Concepts"]
        E1["Topic: Mindfulness"]
        E2["Topic: Legal Contract"]
        E3["Topic: Python Code"]
    end
    subgraph mid["Dims 257-768: Topic Detail"]
        M1["Mindfulness, breath awareness"]
        M2["Contract, IP clause"]
    end
    subgraph tail["Dims 769-3072: Fine-Grained Nuance"]
        T1["Breath awareness vs witnessing"]
        T2["IP clause, indemnity carve-out, SaaS"]
    end

    early --> mid --> tail

    style early fill:#1e3a5f,color:#fff
    style mid fill:#374151,color:#fff
    style tail fill:#5c1a1a,color:#fff
```

Because the dimensions are sorted by importance, you can literally cut off the tail end. Truncating 3,072 to 768 dimensions saves **75% on storage and index size**, with acceptable accuracy loss for broad queries.

### The thing most people miss

This only works if the model was trained with MRL. Apply truncation to a model that wasn't, and you're randomly deleting 25% of its semantic representation. Accuracy will collapse, and it won't be obvious why.

```mermaid
flowchart LR
    Q["Apply truncation to model X"]

    Q -->|"Model trained with MRL"| Safe["Safe\nDimensions are importance-sorted\nTail is genuinely less critical"]
    Q -->|"Standard model, no MRL"| Danger["DANGEROUS\nDimensions are not sorted\nYou are randomly deleting 25% of meaning"]

    style Safe fill:#065f46,stroke:#047857,color:#fff
    style Danger fill:#5c1a1a,stroke:#7c2929,color:#fff
```

**Models that support MRL:**
- `gemini-embedding-002` (truncate to 768, 512, 256, or 128)
- `text-embedding-3-small` and `text-embedding-3-large` (OpenAI)

**Models that don't. Don't truncate these:**
- `BGE-M3`
- `E5-mistral-7b`
- Most Sentence-Transformers pre-2024

### When to use it

- 10M+ vectors with a tight RAM budget
- Your model explicitly documents MRL support
- Queries are broad topic-level retrieval, not fine-grained distinctions
- You've run an eval to confirm accuracy holds before deploying

---

## 5. Path 3: halfvec Quantization

This one is simpler to reason about. Instead of cutting dimensions, we keep all of them and just lower the precision of each number. A 32-bit float like `0.12345678` becomes a 16-bit float like `0.1234`. Same shape, less storage.

```mermaid
flowchart LR
    F32["32-bit float\n4 bytes per dimension\ne.g. 0.12345678"]
    F16["16-bit float halfvec\n2 bytes per dimension\ne.g. 0.1234"]
    SAVE["Result: 50% smaller\nAll dimensions kept\nVector shape preserved"]

    F32 -->|cast precision| F16
    F16 --> SAVE

    style F32 fill:#374151,stroke:#9CA3AF,color:#fff
    style F16 fill:#10B981,stroke:#047857,color:#fff
    style SAVE fill:#065f46,stroke:#047857,color:#fff
```

`pgvector` ships this as a native column type: `halfvec`. Full HNSW indexing, up to 4,000 dimensions.

### The constraint that surprises people

Here's something I didn't know until I hit it: `pgvector`'s standard `vector` type has a hard limit on HNSW indexes.

```postgres frame="code" title="schema.sql"
-- Standard vector: HNSW fails at > 2000 dims
CREATE INDEX ON chunks USING hnsw (gemini_vec vector_ip_ops);  -- ERROR

-- halfvec: HNSW works at any supported dimension
CREATE INDEX ON chunks USING hnsw (gemini_vec halfvec_ip_ops); -- Works
```

If you're using `gemini-embedding-002` (3,072 dims) with a plain `vector` column, you can't build an HNSW index at all. Every query falls back to a sequential scan. `halfvec` isn't optional here. It's the only path that works.

```mermaid
flowchart LR
    DIM["gemini-embedding-002\n3072 dimensions"]

    DIM -->|"store as vector"| FAIL["vector(3072)\nHNSW index: BLOCKED\nFallback: sequential scan\nQuery latency: O(n)"]
    DIM -->|"store as halfvec"| OK["halfvec(3072)\nHNSW index: WORKS\nQuery latency: O(log n)"]

    style FAIL fill:#5c1a1a,stroke:#7c2929,color:#fff
    style OK fill:#065f46,stroke:#047857,color:#fff
```

### When to use it

- Any model without MRL support that still needs memory reduction
- Any 3,072-dim model with `pgvector` (the HNSW constraint forces it)
- Domains where deep semantic nuance matters: legal, philosophical, medical
- When you want the full vector shape but can accept a small precision drop

---

## 6. Side-by-Side Comparison

| Property | No Optimization | Matryoshka Truncation | halfvec Quantization |
|---|---|---|---|
| Memory saving | 0% | Up to 75% | 50% |
| Accuracy impact | None | Low to High (domain-dependent) | Negligible |
| Works with any model | Yes | No, MRL-only | Yes |
| pgvector HNSW on 3072-dim | Blocked | Yes (dims reduced) | Yes |
| Preserves all dimensions | Yes | No | Yes |
| Good for broad queries | Yes | Yes | Yes |
| Good for nuanced queries | Yes | No | Yes |
| Complexity added | None | Low | Low |

---

## 7. The Decision Framework

Here's the tree I walk through when I'm making this call for any new system.

```mermaid
flowchart TD
    S1{"Do you have a hard\nmemory or storage constraint?\n(budget, tier limit, SLA)"}

    S1 -->|No| S2{"Does your vector DB\nsupport HNSW at your\nembedding dimension?"}
    S2 -->|Yes| OUT1["USE NO OPTIMIZATION\n32-bit full vectors\nZero accuracy trade-off"]
    S2 -->|No - HNSW blocked| OUT2

    S1 -->|Yes| S3{"Does your embedding model\nsupport Matryoshka MRL?"}

    S3 -->|No| OUT2["USE HALFVEC\n16-bit, all dimensions\nWorks with any model"]

    S3 -->|Yes| S4{"Does your domain require\ndeep semantic nuance?\ne.g. philosophy, law, medicine"}

    S4 -->|Yes| OUT2
    S4 -->|No - broad topic retrieval| S5{"Have you validated accuracy\nwith a Gold Standard eval?"}

    S5 -->|No - evaluate first| EVAL["RUN EVAL FIRST\nTest truncation vs halfvec\nagainst Gold Queries"]
    S5 -->|Yes - accuracy holds| OUT3["USE MATRYOSHKA TRUNCATION\nTruncate to 768 or lower\nUp to 75% memory saving"]

    EVAL --> S4

    style OUT1 fill:#10B981,stroke:#047857,stroke-width:2px,color:#fff
    style OUT2 fill:#3B82F6,stroke:#1D4ED8,stroke-width:2px,color:#fff
    style OUT3 fill:#F59E0B,stroke:#B45309,stroke-width:2px,color:#fff
    style EVAL fill:#6D28D9,stroke:#4C1D95,stroke-width:2px,color:#fff
```

Two things worth highlighting:

**The eval step is in the tree on purpose.** You shouldn't reach "use Matryoshka" without running a validation pass first. That's the difference between a decision and a guess.

**The pgvector HNSW limit catches people off-guard.** If you pick `gemini-embedding-002` and skip this check, you'll be doing sequential scans in production and wondering why queries are slow.

---

## 8. The Domain Nuance Problem

This is the variable I see underestimated most often, and it's the one that actually hurts people in production.

The same optimization that's perfectly safe for an e-commerce product catalog will quietly destroy retrieval quality for a specialized knowledge corpus. Here's why:

```mermaid
graph TD
    subgraph broad["Broad Domain: E-commerce Product Catalog"]
        B1["Query: red sneakers under 5000 rupees"]
        B2["Chunks: product descriptions, specs, categories"]
        B3["Key signal: in first 256 dims"]
        B1 --> B3
        B2 --> B3
    end

    subgraph narrow["Narrow Domain: Advaita Vedanta Corpus"]
        N1["Query: difference between sakshi and turiya"]
        N2["Chunks: hundreds of paragraphs all mentioning awareness"]
        N3["Key signal: in dims 1500-3072"]
        N1 --> N3
        N2 --> N3
    end

    B3 -->|"Truncation safe"| SAFE["Matryoshka OK\nBroad concepts dominate"]
    N3 -->|"Truncation destroys this"| UNSAFE["halfvec required\nNuance lives in the tail"]

    style SAFE fill:#065f46,stroke:#047857,color:#fff
    style UNSAFE fill:#5c1a1a,stroke:#7c2929,color:#fff
    style broad fill:#1e3a5f,color:#fff
    style narrow fill:#3b1a5f,color:#fff
```

In a narrow domain like the Vedanta corpus above, Matryoshka groups all paragraphs about "awareness" tightly together in the first 768 dimensions. A query about a specific philosophical distinction will pull back whichever paragraph has the most surface-level overlap with the query text, not the semantically correct one.

This isn't an accuracy problem. It's a precision problem. Recall stays high; the system retrieves *something* that looks relevant. Precision collapses; it's just the wrong thing.

---

## 9. What Silent Degradation Actually Looks Like

Most teams don't catch vector optimization errors before production. Here's the actual pattern:

```mermaid
sequenceDiagram
    participant Eng as Engineer
    participant DB as pgvector
    participant User as User

    Eng->>DB: Truncate 3072 to 768 dims, redeploy
    Eng->>Eng: Run 3 manual test queries
    Eng->>Eng: "Looks fine to me"
    Eng->>DB: Ship to production

    User->>DB: "What is the difference between sakshi and turiya?"
    DB->>User: Returns paragraph about generic mindfulness awareness
    User->>User: "This AI doesn't understand my question"
    User->>User: Stops using the product

    Note over Eng,User: No error. No log. No alert. Just silent accuracy degradation.
```

The problem with "run a few queries and check" is that you're testing with broad queries. Those are exactly the queries where truncation works fine. The nuanced ones that break are the ones you didn't think to test.

---

## 10. Prove It: The Gold Standard Eval Loop

Before shipping any optimization, you need one thing: a number. Not a vibe. A measurement that tells you whether the tradeoff is acceptable.

Build this directly into your database schema.

### The schema

```mermaid
erDiagram
    CHUNKS {
        bigint chunk_id PK
        halfvec optimized_vec
        vector raw_vec
        text core_text
    }

    GOLD_QUERIES {
        bigint query_id PK
        text question
        text query_domain
    }

    GOLD_SPANS {
        bigint span_id PK
        bigint query_id FK
        bigint chunk_id FK
        int expected_rank
    }

    EVAL_RUNS {
        bigint run_id PK
        text strategy
        float top1_accuracy
        float top5_accuracy
        float mrr
        timestamptz run_at
    }

    CHUNKS ||--o{ GOLD_SPANS : answered_by
    GOLD_QUERIES ||--o{ GOLD_SPANS : has_ground_truth
    EVAL_RUNS ||--o{ GOLD_QUERIES : evaluated_against
```

### The eval workflow

```mermaid
flowchart TD
    GQ["Gold Queries\n50+ curated questions\nwith known correct answers"]

    GQ --> R1["Run: raw_vec (32-bit)\nRecord retrieval rank\nfor each gold query"]
    GQ --> R2["Run: optimized_vec (halfvec)\nRecord retrieval rank\nfor each gold query"]
    GQ --> R3["Run: truncated_vec (768-dim)\nRecord retrieval rank\nfor each gold query"]

    R1 --> Compare["Compare:\ntop-1 accuracy\ntop-5 accuracy\nMean Reciprocal Rank"]
    R2 --> Compare
    R3 --> Compare

    Compare -->|"halfvec within 2% of raw"| Use2["Use halfvec in production\nYou have mathematical proof"]
    Compare -->|"halfvec drops 10%+ accuracy"| Use1["Use raw_vec\nMemory cost is worth it"]
    Compare -->|"truncated within 3% of halfvec"| Use3["Use truncation\nWith full eval confidence"]

    style Use1 fill:#374151,stroke:#9CA3AF,color:#fff
    style Use2 fill:#065f46,stroke:#047857,color:#fff
    style Use3 fill:#92400e,stroke:#B45309,color:#fff
```

### What your gold queries need to look like

- **Volume:** At least 50 per domain. 200+ if your corpus is specialized.
- **Coverage:** Mix broad queries ("What is mindfulness?") with precise ones ("What distinguishes turiya from deep sleep in Kashmir Shaivism?"). The precise ones are what expose truncation failures.
- **Ground truth:** Not "probably this chunk." The exact chunk ID, verified by someone who actually knows the domain.

### The decision gate

If truncating drops your top-5 retrieval accuracy from 95% to 60%, the evidence is sitting right in your database. The memory savings aren't worth the lobotomy. But if `halfvec` matches raw accuracy within 2%, you have proof the optimization is free. Take it.

The eval loop turns an architectural guess into a decision you can stand behind.

---

Optimize wisely, measure precisely. Sometimes the best algorithm is no algorithm at all.
