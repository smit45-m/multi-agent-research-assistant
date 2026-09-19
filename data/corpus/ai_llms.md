# Artificial Intelligence & Large Language Models: Technical Reference

## Mixture of Experts (MoE) Routing and Sparse Gating

Mixture of Experts architectures replace dense feed-forward blocks with many
parallel expert networks and a learned router. The router computes token
routing probabilities and applies sparse top-k gating so each token activates
only one or two experts, keeping FLOPs nearly constant while parameter count
grows. Training stability depends on an auxiliary router loss (load balancing
loss) that penalizes uneven expert utilization, and on an expert capacity
factor that bounds how many tokens each expert can accept per batch; overflow
tokens are dropped or passed through residual connections. Switch Transformer
showed top-1 routing with capacity factor 1.25 trains stably, while GShard
used top-2 gating. Trade-offs include communication overhead in
expert-parallel all-to-all dispatch and degraded quality if load balancing
collapses to a few hot experts.

## FlashAttention Memory Bandwidth Optimization

FlashAttention reorganizes exact attention computation around the GPU memory
hierarchy. Standard attention materializes the full N-by-N score matrix in
high-bandwidth memory (HBM), making the operation IO-bound. FlashAttention
uses kernel tiling: it loads query, key, and value blocks into on-chip SRAM,
computes tiled softmax incrementally with the online softmax algorithm
(tracking running row maxima and normalizers), and never writes the full
attention matrix to HBM. This reduces memory traffic from O(N^2) to O(N^2 *
d / M) where M is SRAM size, yielding 2-4x wall-clock speedups and enabling
much longer sequences. IO complexity analysis, not FLOP reduction, explains
the gains: the computation is identical, but data movement dominates cost.

## Direct Preference Optimization (DPO) vs RLHF

RLHF fine-tunes a policy with PPO against a reward model trained on human
preference pairs, requiring four model copies (policy, reference policy,
reward model, value function) and suffering from PPO stability issues such
as reward hacking and KL divergence collapse. DPO removes the explicit
reward model and reinforcement loop: it derives a closed-form implicit
reward from the Bradley-Terry preference model, expressing the optimal
policy directly in terms of a simple cross-entropy classification loss over
preferred and rejected completions, regularized against a frozen reference
policy. DPO is cheaper and more stable, but RLHF with online sampling can
exploit reward models more aggressively and adapts to distribution shift
during training.

## Quantization: INT4, FP4, AWQ, and GPTQ

Post-training quantization compresses LLM weights to 4-bit formats. The core
obstacles are activation outliers — a few channels with magnitudes orders
larger than the rest — which force coarse scaling. Per-channel scaling
assigns a separate scale (and zero-point in asymmetric schemes) to each
output channel, preserving accuracy far better than per-tensor scales. GPTQ
performs second-order weight rounding using an approximate Hessian matrix of
the layer reconstruction loss, quantizing columns iteratively while
compensating remaining weights. AWQ instead identifies salient weight
channels by activation magnitude and rescales them before quantization.
Degradation curves show INT4 typically costs under one point of perplexity
on 7B+ models, with dequantization overhead recovered through reduced memory
bandwidth.

## Multi-Query and Grouped Query Attention (GQA)

Autoregressive decoding is bottlenecked by the key-value cache: every
generated token must read cached K/V tensors for all previous tokens.
Multi-Query Attention (MQA) shares a single K/V head across all query heads,
shrinking the KV cache by the head count factor and improving memory
bandwidth utilization, at some quality cost. Grouped Query Attention (GQA)
interpolates via head sharing: query heads are partitioned into groups, each
group sharing one K/V head. GQA with 8 groups recovers nearly all MQA
inference throughput gains — KV cache compression of 4-8x — while matching
full multi-head quality, which is why Llama-2-70B and later models adopt it.
Meeting a latency SLA at high batch sizes typically hinges on KV cache size
more than on compute.

## Speculative Decoding

Speculative decoding accelerates LLM inference by pairing a small draft
model with the large target model. The draft model proposes a run of k
tokens autoregressively; the target model then scores all k proposals in a
single parallel forward pass (target model verification). A rejection
sampling rule accepts a prefix of the proposals — the acceptance rate
determines the speedup ratio — and guarantees the output distribution
exactly matches the target model alone. Extensions like token trees (Medusa,
SpecInfer) propose branching candidate continuations verified jointly.
Practical speedup ratios of 2-3x are achieved when the draft model agrees
with the target on easy tokens, with no quality loss by construction.

## RAG Chunking Strategies and Hybrid Retrieval

Retrieval-Augmented Generation quality depends heavily on chunking and
retrieval design. Fixed-size chunks with token overlap (e.g. 1000 characters
with 200 overlap) are robust defaults; semantic boundary chunking that
splits on section and paragraph borders preserves coherent units. Hybrid
retrieval combines dense embeddings (semantic similarity via vector search)
with sparse BM25 (exact lexical matching robust to rare terms and IDs), and
merges rankings with reciprocal rank fusion: RRF(d) = sum over rankers of
1/(k + rank(d)), commonly with k=60. Multi-query expansion rewrites the user
question into several paraphrases to broaden recall. Hybrid + RRF typically
beats either retriever alone on heterogeneous corpora.

## Multi-Agent Coordination: CrewAI and LangGraph

Multi-agent LLM frameworks structure complex workflows as coordinated
specialist agents. LangGraph models the workflow as a state machine: nodes
are agent functions over a shared typed state, edges define execution order,
and conditional edges implement feedback loops (e.g. re-retrieval when
confidence is low, or writer revision when a critic rejects a draft).
CrewAI emphasizes role-based agent delegation with sequential or
hierarchical processes and task descriptions per agent. Robust systems add
task decomposition by a planner agent, tool-use by retrieval agents, and
explicit verification stages, with bounded iteration counts to prevent
infinite loops.

## Contextual Compression and Reranking

First-stage retrievers optimize recall; rerankers optimize precision. A
bi-encoder embeds query and passages independently, enabling fast
approximate nearest-neighbor search but limited interaction modeling. A
cross-encoder concatenates query and passage and scores them jointly with
full attention, providing far more accurate relevance scoring at higher
cost, so it is applied only to the top 50-100 candidates for passage
reordering. Contextual compression further trims retrieved passages to the
sentences relevant to the query, conserving the downstream context window.
The standard pipeline is: dense/sparse recall, cross-encoder rerank,
compression, then generation.

## Long-Context Retrieval and RoPE Scaling

Rotary position embedding (RoPE) encodes positions as rotations applied to
query and key vectors, giving relative-position-aware attention. Extending a
pretrained model beyond its training window causes attention decay and
out-of-distribution rotation frequencies. Position interpolation compresses
positions linearly into the trained range, while NTK-aware scaling rescales
RoPE base frequency so high-frequency components extrapolate and
low-frequency components interpolate; YaRN combines both with attention
temperature adjustment. Needle-in-a-haystack tests measure whether a fact
placed at arbitrary depth in a long context can be retrieved; models often
show U-shaped accuracy ("lost in the middle") without targeted long-context
fine-tuning.
