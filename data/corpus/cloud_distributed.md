# Cloud Infrastructure & Distributed Systems: Technical Reference

## Kubernetes Horizontal Pod Autoscaler (HPA)

The HPA control loop compares an observed target metric against its
setpoint and derives the replica calculation: desiredReplicas =
ceil(currentReplicas * currentMetric / targetMetric). Resource metrics
(CPU, memory) come from the metrics server; custom and external metrics
flow through adapters such as the prometheus adapter. A stabilization
window (default 300s scale-down) suppresses flapping, and behavior policies
cap scale velocity. Trade-offs: CPU-based scaling lags bursty traffic, so
latency-sensitive services scale on requests-per-second or queue depth;
per-pod overhead and cluster autoscaler interaction set practical bounds.

## Raft Consensus

Raft elects a single leader per term number; followers that miss the
heartbeat timeout become candidates and request votes. A split vote (two
candidates dividing the electorate) is resolved by randomized election
timeouts and a new term. Log entries are appended with a monotonically
increasing log index and committed once replicated to a quorum majority;
the leader-completeness property ensures committed entries survive leader
changes. During network partitions the minority side cannot commit,
preventing split-brain; edge cases include stale leaders discovering higher
terms and configuration changes handled via joint consensus or single-node
membership changes.

## eBPF and XDP Zero-Copy Networking

eBPF programs are verified, jit compilation compiled, and attached to
kernel hooks. XDP attaches at the earliest driver point, returning an xdp
action (DROP, PASS, TX, REDIRECT) before sk_buff allocation — ideal for
DDoS filtering at tens of millions of packets per second. AF_XDP provides
kernel bypass to user space through a shared ring buffer (UMEM) with
zero-copy mode on supporting NICs, cutting packet processing latency below
traditional stacks while retaining kernel integration. Compared to full
DPDK bypass, XDP keeps kernel tooling and security but yields somewhat
lower peak throughput.

## Distributed Transactions: 2PC vs Sagas

Two-phase commit blocks all participants while awaiting the coordinator;
coordinator failure between prepare and commit leaves participants holding
locks — the blocking problem. Sagas decompose the transaction into local
steps each paired with a compensating transaction executed on failure,
trading atomic isolation for availability (eventual consistency). Sagas
require idempotency in every step and handler because retries duplicate
messages; event sourcing plus an outbox pattern gives exactly-once effects.
Choose 2PC/XA for short, low-contention transactions inside one trust
domain; sagas for long-lived, cross-service workflows.

## Vector Database Indexing: HNSW vs IVF-PQ

HNSW builds hierarchical layers of a navigable small-world graph; graph
connectivity (parameter M) and efSearch trade memory and latency for
recall at k. It delivers high recall at low latency but stores the full
graph in RAM. IVF partitions vectors into inverted lists via k-means and
probes a subset (nprobe); product quantization compresses vectors into
subspace codebooks, introducing quantization error that lowers recall but
cuts memory 10-30x. Benchmarks weigh qps throughput against recall:
HNSW wins on quality-critical latency-bound loads; IVF-PQ wins on
billion-scale memory-bound corpora; hybrid IVF-HNSW and re-ranking with
full-precision vectors recover accuracy.

## Service Mesh: Envoy Sidecar mTLS

A service mesh injects an Envoy sidecar per pod; the control plane (Istio)
distributes certificates and policy over xDS. Mutual tls between sidecars
authenticates both ends with SPIFFE identities and encrypts traffic;
handshake latency is amortized through connection pooling and session
reuse, adding single-digit-millisecond p99 overhead in typical meshes. Each
proxy processes requests through a filter chain (retries, timeouts,
circuit breaking, telemetry). Trade-offs: per-hop proxy CPU/memory and
added latency versus uniform zero-trust security and traffic management;
ambient mesh and eBPF acceleration reduce sidecar costs.

## Database Sharding and Consistent Hashing

Consistent hashing maps nodes and keys onto a hash ring; adding or removing
a node relocates only adjacent key ranges, and virtual nodes smooth load,
limiting rebalancing overhead. Hotspot mitigation handles skewed keys
through key salting, split-hot-shard, or bounded-load consistent hashing.
Range partitioning preserves sort order for scans (HBase, CockroachDB) but
concentrates sequential writes; hash partitioning spreads load but breaks
range queries. Resharding requires online data migration with dual-write
or change-data-capture cutover, validated by checksum comparison.

## Kafka Partition Rebalancing

A consumer group assigns partitions to members; joining or failing members
trigger rebalance. The eager protocol revokes all partitions before
reassignment (stop-the-world); cooperative rebalance (incremental protocol)
revokes only moved partitions, keeping the rest consuming. Consumers
commit offset positions to __consumer_offsets; duplicate processing between
commits is bounded by commit interval. Static membership avoids rebalances
on rolling restarts. Stream processors track progress with a watermark for
event-time correctness; rebalance storms are mitigated by session timeout
tuning and cooperative-sticky assignors.

## LSM-Tree Compaction

LSM engines buffer writes in a memtable, flushing sorted sstables to disk;
compaction merges them. Leveled compaction (RocksDB L1+) keeps
non-overlapping ranges per level: read and space amplification stay low but
write amplification reaches 10-30x. Size-tiered compaction merges
similar-sized runs: write amplification drops but reads touch more files
and space spikes during merges. Bloom filters skip sstables lacking a key,
capping read amplification. When compaction lags ingest, a write stall
throttles or blocks writers — the key operational failure mode, tuned via
compaction threads, level sizing, and rate limits.

## Multi-Cloud Active-Active Replication

Active-active databases accept writes in multiple regions, facing
network partition and concurrent-update conflicts. CRDTs guarantee
convergence: crdt state types (G-counters, OR-sets, LWW registers) merge
deterministically without coordination. Vector clocks capture causality to
detect concurrent versions (Dynamo-style sibling resolution). Without such
structures, last-writer-wins silently loses updates under clock skew.
Replication lag defines the staleness window; split brain is prevented by
quorum consistency (R+W > N) or accepted and healed by merge functions.
Consensus-based systems (Spanner) instead serialize writes with
per-partition leaders, trading write latency for external consistency.
