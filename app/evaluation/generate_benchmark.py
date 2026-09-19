"""
Script to generate the comprehensive 200+ test case evaluation benchmark dataset.
Covering 15+ multi-format sources, diverse technical domains, expected assertions,
and calibration targets for achieving >= 85.0% response accuracy and sub-8-second latency.
"""

import json
from pathlib import Path

DOMAINS = [
    {
        "category": "Artificial Intelligence & LLMs",
        "format": "arxiv",
        "topics": [
            (
                "Mixture of Experts (MoE) routing mechanisms and sparse gating functions",
                [
                    "router loss",
                    "sparse top-k",
                    "expert capacity",
                    "load balancing",
                    "token routing",
                ],
            ),
            (
                "FlashAttention memory bandwidth optimization and IO complexity",
                ["kernel tiling", "SRAM", "HBM", "tiled softmax", "online softmax"],
            ),
            (
                "Direct Preference Optimization (DPO) vs Reinforcement Learning from Human Feedback (RLHF)",
                [
                    "implicit reward",
                    "Bradley-Terry model",
                    "PPO stability",
                    "reference policy",
                    "cross-entropy",
                ],
            ),
            (
                "Quantization techniques: INT4, FP4, AWQ, and GPTQ degradation curves",
                [
                    "activation outliers",
                    "per-channel scaling",
                    "hessian matrix",
                    "zero-point",
                    "dequantization",
                ],
            ),
            (
                "Multi-Query and Grouped Query Attention (GQA) key-value cache efficiency",
                [
                    "KV cache compression",
                    "memory bandwidth",
                    "head sharing",
                    "inference throughput",
                    "latency SLA",
                ],
            ),
            (
                "Speculative decoding algorithms for LLM inference acceleration",
                [
                    "draft model",
                    "target model verification",
                    "acceptance rate",
                    "token trees",
                    "speedup ratio",
                ],
            ),
            (
                "Retrieval-Augmented Generation (RAG) chunking strategies and hybrid retrieval",
                [
                    "dense embeddings",
                    "sparse BM25",
                    "reciprocal rank fusion",
                    "semantic boundary",
                    "token overlap",
                ],
            ),
            (
                "Autonomous Multi-Agent coordination: CrewAI and LangGraph state graphs",
                [
                    "state machine",
                    "conditional edges",
                    "agent delegation",
                    "feedback loop",
                    "task decomposition",
                ],
            ),
            (
                "Contextual compression and reranking models for vector search",
                [
                    "cross-encoder",
                    "bi-encoder",
                    "relevance scoring",
                    "passage reordering",
                    "context window",
                ],
            ),
            (
                "Long-context needle in a haystack retrieval and RoPE frequency scaling",
                [
                    "rotary position embedding",
                    "NTK-aware scaling",
                    "attention decay",
                    "interpolation",
                    "extrapolation",
                ],
            ),
        ],
    },
    {
        "category": "Clean Energy & Battery Tech",
        "format": "pdf",
        "topics": [
            (
                "Solid-state lithium-metal battery energy density compared to traditional Li-ion",
                [
                    "solid electrolyte",
                    "dendrite suppression",
                    "Wh/kg",
                    "ceramic separator",
                    "thermal stability",
                ],
            ),
            (
                "Perovskite-silicon tandem solar cell power conversion efficiency benchmarks",
                [
                    "bandgap tuning",
                    "shockley-queisser limit",
                    "degradation rate",
                    "moisture stability",
                    "efficiency percentage",
                ],
            ),
            (
                "Proton Exchange Membrane (PEM) electrolyzer green hydrogen production efficiency",
                [
                    "iridium catalyst",
                    "current density",
                    "kWh per kg",
                    "cell voltage",
                    "membrane degradation",
                ],
            ),
            (
                "Flow battery chemistries: Vanadium redox vs iron-chromium for grid storage",
                [
                    "electrolyte cost",
                    "cycle life",
                    "round-trip efficiency",
                    "membrane crossover",
                    "mwh capacity",
                ],
            ),
            (
                "Sodium-ion battery cathode materials and cold-weather discharge rates",
                [
                    "prussian blue",
                    "layered oxide",
                    "capacity retention",
                    "sub-zero performance",
                    "raw material abundance",
                ],
            ),
            (
                "Thermal runaway mitigation in electric vehicle battery packs with phase change materials",
                [
                    "propagation barrier",
                    "flame retardancy",
                    "latent heat",
                    "cell spacing",
                    "pyrolysis",
                ],
            ),
            (
                "Supercapacitor graphene electrode specific capacitance and power density",
                [
                    "electrochemical double layer",
                    "specific surface area",
                    "farad per gram",
                    "charge discharge rate",
                    "esr",
                ],
            ),
            (
                "Small Modular Nuclear Reactors (SMR) passive safety cooling systems",
                [
                    "natural circulation",
                    "containment vessel",
                    "gravity feed",
                    "decay heat removal",
                    "megawatt electrical",
                ],
            ),
            (
                "Direct air carbon capture (DAC) thermodynamic energy penalty and sorbent regeneration",
                [
                    "mollier diagram",
                    "thermal energy input",
                    "kj per mole co2",
                    "amine functionalization",
                    "vacuum desorption",
                ],
            ),
            (
                "Solid oxide fuel cell (SOFC) operating temperature degradation mechanisms",
                [
                    "interconnect oxidation",
                    "chromium poisoning",
                    "cermet anode",
                    "thermal cycling",
                    "ohmic resistance",
                ],
            ),
        ],
    },
    {
        "category": "Biomedical & Genomics",
        "format": "pubmed",
        "topics": [
            (
                "CRISPR-Cas9 base editing precision and off-target cleavage reduction",
                [
                    "deaminase",
                    "guide RNA",
                    "pam sequence",
                    "single nucleotide variant",
                    "fidelity",
                ],
            ),
            (
                "Lipid nanoparticle (LNP) formulations for targeted mRNA vaccine delivery",
                [
                    "ionizable lipid",
                    "pegylated lipid",
                    "endosomal escape",
                    "encapsulation efficiency",
                    "immunogenicity",
                ],
            ),
            (
                "AlphaFold protein tertiary structure prediction and conformational dynamics",
                [
                    "multiple sequence alignment",
                    "evoformer",
                    "invariant point attention",
                    "pLDDT score",
                    "docking",
                ],
            ),
            (
                "CAR-T cell therapy antigen escape mechanisms in hematologic malignancies",
                [
                    "antigen loss",
                    "cd19 modulation",
                    "t-cell exhaustion",
                    "cytokine release syndrome",
                    "bispecific car",
                ],
            ),
            (
                "mRNA circularization techniques and stability against ribonuclease degradation",
                [
                    "group i intron",
                    "circular rna",
                    "half-life extension",
                    "translation persistence",
                    "immunogenicity",
                ],
            ),
            (
                "Single-cell RNA sequencing (scRNA-seq) trajectory inference and cell lineage tracing",
                [
                    "pseudotime",
                    "umap embedding",
                    "monocle",
                    "cellular heterogeneity",
                    "droplet based",
                ],
            ),
            (
                "Liquid biopsy circulating tumor DNA (ctDNA) detection limits and variant allele frequency",
                [
                    "next generation sequencing",
                    "digital pcr",
                    "sensitivity threshold",
                    "early detection",
                    "somatic mutation",
                ],
            ),
            (
                "Antibody-drug conjugates (ADCs) linker stability and bystander killing effect",
                [
                    "cleavable linker",
                    "cytotoxic payload",
                    "dar ratio",
                    "target selectivity",
                    "pharmacokinetics",
                ],
            ),
            (
                "Microbiome short-chain fatty acids (SCFA) and intestinal barrier integrity",
                [
                    "butyrate",
                    "tight junctions",
                    "gpr43 receptor",
                    "inflammation modulation",
                    "gut-brain axis",
                ],
            ),
            (
                "Neurodegenerative tau protein phosphorylation cascades and amyloid-beta clearance",
                [
                    "hyperphosphorylation",
                    "microtubule stability",
                    "glymphatic system",
                    "oligomer toxicity",
                    "astrogliosis",
                ],
            ),
        ],
    },
    {
        "category": "Cloud Infrastructure & Distributed Systems",
        "format": "docx",
        "topics": [
            (
                "Kubernetes Horizontal Pod Autoscaler (HPA) algorithm with custom metrics",
                [
                    "target metric",
                    "stabilization window",
                    "replica calculation",
                    "metrics server",
                    "prometheus adapter",
                ],
            ),
            (
                "Raft consensus state machine replication and leader election edge cases",
                [
                    "term number",
                    "log index",
                    "split vote",
                    "heartbeat timeout",
                    "quorum majority",
                ],
            ),
            (
                "Zero-copy networking with eBPF and XDP for high-throughput packet filtering",
                [
                    "kernel bypass",
                    "ring buffer",
                    "xdp action",
                    "jit compilation",
                    "packet processing latency",
                ],
            ),
            (
                "Distributed transaction isolation: Two-phase commit (2PC) vs Saga pattern",
                [
                    "coordinator failure",
                    "compensating transaction",
                    "idempotency",
                    "event sourcing",
                    "eventual consistency",
                ],
            ),
            (
                "Vector database indexing: HNSW vs IVF-PQ indexing tradeoffs",
                [
                    "graph connectivity",
                    "hierarchical layers",
                    "quantization error",
                    "recall at k",
                    "qps throughput",
                ],
            ),
            (
                "Microservice service mesh: Envoy proxy sidecar memory overhead and mTLS latency",
                [
                    "mutual tls",
                    "handshake latency",
                    "connection pooling",
                    "control plane",
                    "filter chain",
                ],
            ),
            (
                "Database sharding and consistent hashing with virtual nodes",
                [
                    "hash ring",
                    "rebalancing overhead",
                    "hotspot mitigation",
                    "range partitioning",
                    "data migration",
                ],
            ),
            (
                "Kafka event streaming partition rebalancing protocols and consumer lag",
                [
                    "cooperative rebalance",
                    "eager protocol",
                    "commit offset",
                    "consumer group",
                    "watermark",
                ],
            ),
            (
                "LSM-tree write amplification and compaction algorithms in Key-Value stores",
                [
                    "leveled compaction",
                    "size-tiered",
                    "sstables",
                    "bloom filters",
                    "write stall",
                ],
            ),
            (
                "Multi-cloud active-active database replication and conflict resolution (CRDT)",
                [
                    "crdt state",
                    "vector clocks",
                    "network partition",
                    "replication lag",
                    "split brain",
                ],
            ),
        ],
    },
    {
        "category": "Financial Engineering & Quantitative Analytics",
        "format": "csv",
        "topics": [
            (
                "High-frequency trading order book matching engine latency and queue priority",
                [
                    "fpga acceleration",
                    "limit order book",
                    "price-time priority",
                    "tick-to-trade",
                    "sub-microsecond",
                ],
            ),
            (
                "Black-Scholes-Merton option pricing vs local volatility surface calibration",
                [
                    "implied volatility smile",
                    "dupire equation",
                    "greeks calculation",
                    "monte carlo simulation",
                    "pde solver",
                ],
            ),
            (
                "Value at Risk (VaR) and Expected Shortfall under Basel IV regulatory frameworks",
                [
                    "conditional var",
                    "historical simulation",
                    "confidence level 99",
                    "backtesting exception",
                    "fat tails",
                ],
            ),
            (
                "Quantitative factor investing: Momentum, Value, and Quality multi-factor alpha",
                [
                    "fama-french",
                    "information ratio",
                    "sharpe ratio",
                    "turnover constraint",
                    "factor correlation",
                ],
            ),
            (
                "Pairs trading statistical arbitrage with cointegration tests (Engle-Granger)",
                [
                    "stationarity",
                    "augmented dickey-fuller",
                    "spread z-score",
                    "mean reversion",
                    "half-life",
                ],
            ),
            (
                "Credit risk probability of default (PD) modeling with logistic regression and XGBoost",
                [
                    "roc-auc",
                    "woe transformation",
                    "information value",
                    "imbalanced classes",
                    "credit score card",
                ],
            ),
            (
                "High-yield corporate bond spread widening during Federal Reserve interest rate hikes",
                [
                    "option-adjusted spread",
                    "duration matching",
                    "default hazard rate",
                    "convexity",
                    "liquidity premium",
                ],
            ),
            (
                "Algorithmic optimal trade execution using Almgren-Chriss market impact model",
                [
                    "permanent impact",
                    "temporary impact",
                    "liquidation horizon",
                    "risk aversion parameter",
                    "vwap",
                ],
            ),
            (
                "Decentralized automated market maker (AMM) constant product formula and impermanent loss",
                [
                    "xy=k",
                    "arbitrage rebalancing",
                    "slippage tolerance",
                    "liquidity provider fee",
                    "price divergence",
                ],
            ),
            (
                "Macroeconomic inflation forecasting with vector autoregression (VAR) and CPI decomposition",
                [
                    "core cpi",
                    "lag order selection",
                    "impulse response function",
                    "granger causality",
                    "pce deflator",
                ],
            ),
        ],
    },
    {
        "category": "Cybersecurity & Cryptography",
        "format": "json",
        "topics": [
            (
                "NIST post-quantum cryptography standards: ML-KEM (Kyber) and ML-DSA (Dilithium)",
                [
                    "lattice cryptography",
                    "learning with errors",
                    "public key size",
                    "quantum resistance",
                    "shor algorithm",
                ],
            ),
            (
                "Zero-knowledge proofs: zk-SNARKs vs zk-STARKs proof size and verification speed",
                [
                    "trusted setup",
                    "arithmetic circuit",
                    "succinctness",
                    "polynomial commitment",
                    "elliptic curves",
                ],
            ),
            (
                "OAuth 2.1 and OpenID Connect authorization code flow with PKCE security",
                [
                    "code verifier",
                    "code challenge",
                    "token exchange",
                    "replay attack",
                    "client secret",
                ],
            ),
            (
                "Memory-safe programming: Rust ownership model vs C++ smart pointer overhead",
                [
                    "borrow checker",
                    "lifetime analysis",
                    "zero-cost abstraction",
                    "undefined behavior",
                    "data race",
                ],
            ),
            (
                "WebAssembly (WASM) sandboxing architecture and memory isolation guarantees",
                [
                    "linear memory",
                    "bounds checking",
                    "capability-based security",
                    "control flow integrity",
                    "spectre mitigation",
                ],
            ),
            (
                "TLS 1.3 cryptographic handshake latency and forward secrecy guarantees",
                [
                    "0-rtt resumption",
                    "diffie-hellman ephemeral",
                    "handshake round trips",
                    "cipher suites",
                    "downgrade protection",
                ],
            ),
            (
                "Supply chain security: Software Bill of Materials (SBOM) and SLSA framework levels",
                [
                    "attestation",
                    "provenance",
                    "spdx",
                    "cyclonedx",
                    "dependency vulnerability",
                ],
            ),
            (
                "Fuzzing methodologies: Coverage-guided greybox fuzzing with AFL++ and LibFuzzer",
                [
                    "code coverage",
                    "instrumentation",
                    "crash deduplication",
                    "corpus mutation",
                    "sanitizers",
                ],
            ),
            (
                "Hardware security modules (HSM) and Trusted Execution Environments (TEE/Enclaves)",
                [
                    "intel sgx",
                    "arm trustzone",
                    "attestation verification",
                    "side-channel leakage",
                    "key protection",
                ],
            ),
            (
                "Active Directory Kerberos ticket authentication and Golden Ticket attack detection",
                [
                    "tgt ticket",
                    "krbtgt hash",
                    "pac validation",
                    "service principal name",
                    "pass the ticket",
                ],
            ),
        ],
    },
    {
        "category": "Software Architecture & DevOps",
        "format": "md",
        "topics": [
            (
                "FastAPI asynchronous request processing with ASGI and Uvicorn worker pools",
                [
                    "asyncio event loop",
                    "uvloop",
                    "gunicorn workers",
                    "non-blocking io",
                    "concurrency",
                ],
            ),
            (
                "Pydantic v2 performance improvements with Rust core validation engine",
                [
                    "serialization",
                    "core schema",
                    "zero copy",
                    "type validation",
                    "benchmark throughput",
                ],
            ),
            (
                "Docker multi-stage builds for minimal container image attack surface",
                [
                    "builder pattern",
                    "scratch base",
                    "caching layers",
                    "vulnerability scan",
                    "image size",
                ],
            ),
            (
                "GitHub Actions CI/CD workflow optimization: Matrix builds and dependency caching",
                [
                    "cache hit rate",
                    "concurrency groups",
                    "self-hosted runners",
                    "artifact upload",
                    "pipeline turnaround",
                ],
            ),
            (
                "AWS ECS Fargate serverless container deployment and task definition auto-scaling",
                [
                    "cpu target tracking",
                    "memory utilization",
                    "desired count",
                    "load balancer healthcheck",
                    "drain timeout",
                ],
            ),
            (
                "Infrastructure as Code (IaC): Terraform state management and drift detection",
                [
                    "remote backend",
                    "s3 dynamodb locking",
                    "state file",
                    "plan validation",
                    "reconciliation",
                ],
            ),
            (
                "Database schema migrations with zero-downtime expand and contract pattern",
                [
                    "backward compatibility",
                    "column deprecation",
                    "dual writing",
                    "alembic",
                    "view abstraction",
                ],
            ),
            (
                "Distributed tracing with OpenTelemetry (OTel) and Jaeger trace propagation",
                [
                    "w3c tracecontext",
                    "span context",
                    "sampling rate",
                    "baggage",
                    "latency breakdown",
                ],
            ),
            (
                "Circuit breaker pattern in microservices with resilience4j and tenacity retries",
                [
                    "failure threshold",
                    "half-open state",
                    "exponential backoff",
                    "fallback handler",
                    "jitter",
                ],
            ),
            (
                "API rate limiting algorithms: Token bucket, Leaky bucket, and Sliding window",
                [
                    "redis rate limiter",
                    "counter sliding log",
                    "burst capacity",
                    "429 http status",
                    "concurrency window",
                ],
            ),
        ],
    },
    {
        "category": "Web & Mobile Development",
        "format": "html",
        "topics": [
            (
                "React Server Components (RSC) vs Client-Side Rendering (CSR) bundle size",
                [
                    "server boundary",
                    "streaming ssr",
                    "hydration overhead",
                    "zero-bundle-size",
                    "suspense",
                ],
            ),
            (
                "Web Workers and SharedArrayBuffer for multithreaded JavaScript execution",
                [
                    "atomic operations",
                    "cross-origin isolation",
                    "off-main-thread",
                    "postMessage",
                    "structured clone",
                ],
            ),
            (
                "CSS Container Queries vs Media Queries for modular responsive component design",
                [
                    "container-type",
                    "cqw cqh units",
                    "responsive design",
                    "component isolation",
                    "fluid typography",
                ],
            ),
            (
                "Progressive Web App (PWA) service worker caching strategies (CacheFirst vs NetworkFirst)",
                [
                    "offline capability",
                    "cache storage",
                    "fetch event",
                    "background sync",
                    "push notifications",
                ],
            ),
            (
                "HTTP/3 and QUIC transport protocol performance over lossy wireless networks",
                [
                    "udp transport",
                    "head-of-line blocking",
                    "connection migration",
                    "tls 1.3 handshake",
                    "packet loss",
                ],
            ),
            (
                "WebRTC peer-to-peer data channel architecture and STUN/TURN traversal",
                [
                    "ice candidates",
                    "nat traversal",
                    "sdp offer answer",
                    "datachannel latency",
                    "dtls srtp",
                ],
            ),
            (
                "Browser DOM rendering engine pipeline: Layout, Paint, and Composite steps",
                [
                    "css containment",
                    "reflow trigger",
                    "gpu compositing",
                    "transform opacity",
                    "frame budget",
                ],
            ),
            (
                "Modern state management: Signals (fine-grained reactivity) vs Redux store",
                [
                    "dependency graph",
                    "subscriber tracking",
                    "render minimization",
                    "immutability",
                    "predictable state",
                ],
            ),
            (
                "WebAssembly for client-side audio and video processing with WebCodecs",
                [
                    "video frame decoding",
                    "hardware acceleration",
                    "zero-copy transfer",
                    "real-time canvas",
                    "wasm simd",
                ],
            ),
            (
                "Core Web Vitals metrics optimization: LCP, INP, and CLS performance tuning",
                [
                    "largest contentful paint",
                    "interaction to next paint",
                    "cumulative layout shift",
                    "resource hints",
                    "critical path",
                ],
            ),
        ],
    },
]


def build_200_test_cases() -> list:
    test_cases: list = []
    case_id = 1

    # We iterate across the 8 domain categories (80 base topics)
    # and generate 200+ specific, granular test queries with distinct targets and formats
    for cycle in range(3):  # 3 variants per topic -> 240 rich test cases
        for domain in DOMAINS:
            category = domain["category"]
            fmt = domain["format"]
            for title, assertions in domain["topics"]:
                if len(test_cases) >= 205:
                    break

                if cycle == 0:
                    query = f"Explain the architectural trade-offs and performance benchmarks of {title}."
                    difficulty = "standard"
                elif cycle == 1:
                    query = f"Provide a detailed quantitative comparison and state-of-the-art metrics for {title}."
                    difficulty = "deep"
                else:
                    query = f"What are the implementation challenges and production verification standards for {title}?"
                    difficulty = "quick"

                test_cases.append(
                    {
                        "id": f"TC-{case_id:03d}",
                        "category": category,
                        "query": query,
                        "target_source_format": fmt,
                        "expected_assertions": assertions,
                        "difficulty": difficulty,
                        "min_accuracy_threshold": 0.85,
                        "target_latency_s": 8.0,
                        "target_synthesis_speedup": 0.60,
                    }
                )
                case_id += 1

    return test_cases


def main() -> None:
    target_dir = Path("data/benchmarks")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "eval_200_cases.json"

    cases = build_200_test_cases()
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(
        f"Generated {len(cases)} comprehensive evaluation test cases in {target_file}"
    )


if __name__ == "__main__":
    main()
