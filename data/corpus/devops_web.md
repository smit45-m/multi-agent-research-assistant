# Software Architecture, DevOps, Web & Mobile: Technical Reference

## FastAPI Async Request Processing

FastAPI on uvicorn runs an asyncio event loop; uvloop (libuv-based)
roughly doubles loop throughput versus the stdlib loop. Async handlers
must await non-blocking io — an accidental blocking call stalls every
request on the loop, the classic pitfall. Sync handlers are dispatched to
a bounded thread pool instead. Production deploys multiple gunicorn
workers (or uvicorn --workers) to use all cores, sized roughly per-core
for CPU-bound and higher for IO-bound loads. concurrency is bounded
deliberately with semaphores/capacity limiters so bursts degrade to
queueing rather than thread oversubscription; benchmarks show async wins
chiefly on high-fanout IO workloads.

## Pydantic v2 Performance

Pydantic v2 rebuilt validation in Rust (pydantic-core): the Python layer
compiles models into a core schema executed natively, delivering 5-50x
faster type validation than v1. serialization also moves to Rust with
direct-to-JSON paths that avoid intermediate dicts; benchmark throughput
on typical API models shows several-fold end-to-end gains in request
handling. Strict mode removes lossy coercion, discriminated unions
short-circuit matching. TypeAdapter validates non-model types with zero
copy where possible into Python objects; the practical impact is that
validation stops being the API bottleneck.

## Docker Multi-Stage Builds

Multi-stage builds implement the builder pattern: a heavy stage compiles
dependencies; the runtime stage copies only artifacts, shrinking image
size (a Python app drops from ~1GB single-stage to 100-200MB slim, and Go
binaries run from a scratch base at single-digit MB). Layer ordering
drives caching layers efficiency: copy dependency manifests and install
before copying source so code edits do not bust the dependency cache
(BuildKit cache mounts help further). Smaller images shrink the
vulnerability scan surface (Trivy, Grype in CI), speed pulls and
autoscaling cold starts, and pin exact base digests for reproducibility.

## GitHub Actions Optimization

Pipeline turnaround shortens through parallel jobs and dependency caching:
actions/cache keyed on lockfile hashes: a high cache hit rate saves
minutes per run (setup-python/setup-node have built-in cache modes).
concurrency groups cancel superseded runs on force-push, saving runner
minutes. Matrix builds fan out test shards; artifact upload passes builds
between jobs rather than rebuilding. self-hosted runners give custom
hardware and warm caches but require patching and security isolation for
untrusted PRs. Split fast lint/unit from slow integration stages so
feedback lands early; path filters skip unaffected workflows.

## ECS Fargate Auto-Scaling

Fargate runs containers serverless; service auto scaling adjusts
desired count via cpu target tracking (e.g. hold 60 percent) and
memory utilization policies, or step/scheduled scaling for known peaks.
The load balancer healthcheck gates rollouts: unhealthy targets are
deregistered and wait out the drain timeout (default 300s, tunable) for
connection draining before task stop. Right-sizing task CPU/memory
combinations controls cost (Fargate bills per vCPU-second and GB-second);
Spot capacity cuts cost ~70 percent for interruption-tolerant services.
Cold-start latency for new tasks (image pull, ENI attach) sets a floor on
scale-out response, mitigated by headroom or predictive scaling.

## Terraform State Management

Terraform state file maps configuration to real resources; teams use a
remote backend — canonically s3 dynamodb locking (state in S3, lock table
in DynamoDB) — to serialize applies and prevent corruption (native S3
lockfile support arrives in 1.10+). plan validation in CI (fmt, validate,
tflint, OPA/Sentinel policy) reviews the diff before apply.
reconciliation handles drift: terraform plan -refresh-only detects
out-of-band changes; import adopts unmanaged resources. State contains
secrets — encrypt at rest, restrict access, never commit. Workspaces or
directory-per-environment isolate blast radius; module pinning keeps
upgrades deliberate.

## Zero-Downtime Schema Migrations

The expand-and-contract pattern preserves backward compatibility: expand
(add nullable columns, new tables, indexes CONCURRENTLY), migrate code to
dual writing old and new schema, backfill in batches, flip reads, then
contract via column deprecation and delayed drops. Never rename in place
— old app versions run during rollout. alembic (or Flyway/Liquibase)
versions migrations with up/down scripts in CI. A view abstraction can
decouple app queries from physical tables during transitions. Guard
against lock storms: ALTER TABLE taking ACCESS EXCLUSIVE on hot tables,
long backfills without batching, and missing statement timeouts.

## Distributed Tracing with OpenTelemetry

OpenTelemetry standardizes traces, metrics, and logs. Context propagates
via w3c tracecontext headers (traceparent); span context carries trace ID
and span ID across service hops, while baggage carries key-values for
cross-cutting concerns (sparingly — it rides every request). A sampling
rate balances cost and visibility: head sampling is cheap; tail sampling
keeps errors and slow traces at the collector after seeing the whole
trace. The trace waterfall gives a latency breakdown per hop, exposing
fanout amplification, serial chains, and retry storms; exemplars link
metrics spikes directly to offending traces.

## Circuit Breaker Pattern

A circuit breaker trips after a failure threshold (error rate or
consecutive failures in a rolling window), fast-failing calls to a sick
dependency instead of queueing threads. After a cooldown it enters the
half-open state, admitting probe requests: success closes, failure
re-opens. Every call path defines a fallback handler — cached data,
default response, or explicit degradation. Retries pair with exponential
backoff plus jitter to prevent synchronized retry storms (thundering
herd); retry budgets cap amplification. Bulkheads isolate resource pools
per dependency so one failure cannot exhaust shared capacity: the
combination converts cascading failures into contained brownouts.

## API Rate Limiting Algorithms

Token bucket refills at rate r with burst capacity b — allowing short
bursts while enforcing average rate — and answers in O(1); it is the
default for API gateways. Fixed windows suffer boundary spikes (2x at
edges); a sliding window (or counter sliding log hybrid) smooths this,
with sliding logs exact but memory-heavy per client. Distributed
enforcement uses a redis rate limiter — atomic Lua scripts or
INCR-with-TTL — trading a network hop for global consistency; local
buckets with async sync trade small overshoot for latency. Rejections
return 429 http status with Retry-After; concurrency window limits
(inflight caps) complement rate limits for expensive endpoints.

## React Server Components

RSC splits the tree at the server boundary: server components render to a
serialized element stream with zero-bundle-size cost — their code and
dependencies never ship to the client — while client components hydrate
interactively. streaming ssr with suspense sends HTML progressively as
data resolves, cutting time-to-first-byte and largest contentful paint.
hydration overhead falls because only client islands hydrate. Trade-offs:
server boundaries constrain prop serialization, mental model complexity
rises, and mutations flow through server actions; frameworks (Next.js App
Router) manage the RSC payload protocol and caching layers.

## Web Workers and SharedArrayBuffer

Web Workers move computation off-main-thread; postMessage transfers data
via structured clone (deep copy) or transfers ownership of Transferables
(ArrayBuffer) at zero copy cost. SharedArrayBuffer gives true shared
memory across threads with atomic operations (Atomics.add, wait/notify)
for lock-free coordination — required for multithreaded WASM (pthreads).
Spectre concerns gate SAB behind cross-origin isolation: COOP and COEP
headers isolate the browsing context before SAB and high-resolution
timers are enabled. Rule of thumb: batch messages, prefer transfer over
clone for large buffers, and keep main-thread work under the frame
budget.

## CSS Container Queries

Media queries respond to the viewport; container queries respond to an
ancestor container's size, enabling true component isolation — the same
card adapts whether placed in a sidebar or main column. An element opts
in with container-type: inline-size (or size), becoming a query
container; @container rules then style descendants by its width. cqw cqh
units (1 percent of container width/height) enable fluid typography
scaled to the component, not the viewport. This shifts responsive design
from page-level breakpoints to self-contained components, composing
safely in design systems; nested named containers target specific
ancestors.

## PWA Service Workers

A service worker is a network proxy: the fetch event intercepts requests
and applies caching strategies via cache storage — cache-first for
static assets, stale-while-revalidate for semi-fresh data, network-first
for API calls — yielding offline capability with an app-shell fallback.
background sync queues failed writes for replay when connectivity
returns; periodic sync refreshes content. push notifications arrive via
Web Push (VAPID) and display even with the page closed. Lifecycle
gotchas: waiting workers activate only after old tabs close unless
skipWaiting, and cache versioning must purge stale entries on activate.

## HTTP/3 and QUIC

HTTP/3 runs on QUIC over udp transport with TLS 1.3 built in: the
combined transport+crypto handshake completes in one RTT (0-RTT for
resumption), versus TCP+TLS's two-plus. QUIC eliminates transport-level
head-of-line blocking: packet loss in one stream never stalls others,
unlike HTTP/2 where one lost TCP segment blocks all multiplexed streams —
the biggest win on lossy mobile networks. connection migration keeps
sessions alive across network switches (Wi-Fi to cellular) via connection
IDs instead of the IP 4-tuple. Costs: user-space processing overhead and
UDP middlebox friction, so clients race H3 with H2 fallback (Alt-Svc).

## WebRTC Data Channels

WebRTC negotiates peer connections through sdp offer answer exchange over
an application signaling channel. nat traversal uses ICE: agents gather
ice candidates (host, STUN server-reflexive, TURN relay) and probe pairs
for connectivity — TURN relays as fallback when symmetric NATs block
direct paths (10-20 percent of connections). Media encrypts via dtls srtp;
data channels run SCTP over DTLS with configurable reliability: unordered
and maxRetransmits=0 approximates UDP for games, ordered-reliable behaves
like TCP. datachannel latency on direct paths beats relayed WebSocket
round trips, which is the point for real-time apps.

## Browser Rendering Pipeline

The pipeline runs style, layout, paint, and composite within a ~16ms
frame budget at 60fps. Layout-affecting property changes (width, top)
cause a reflow trigger — synchronous layout recomputation, worst when
forced mid-frame by reading offsetHeight after writes (layout thrashing).
transform opacity changes skip layout and paint entirely, animating on
the compositor thread via gpu compositing of promoted layers — the only
properties that animate cheaply. css containment (contain: layout paint;
content-visibility: auto) fences subtree invalidation and skips
offscreen rendering. Tools: will-change promotes judiciously; too many
layers explode memory.

## Signals vs Redux State Management

Signals (SolidJS, Vue refs, Angular signals, TC39 proposal) are reactive
primitives: reads register subscriber tracking within a dependency graph,
so a write updates precisely the consumers that read it — render
minimization without virtual-DOM diffing or selector memoization. Redux
centralizes state with immutability and pure reducers: every change is an
action through one store, giving predictable state, time-travel
debugging, and simple persistence at the cost of boilerplate and
selector-based re-render tuning. Signals win on fine-grained update
performance; Redux-style stores win on auditability of complex shared
workflows — modern apps combine both.

## WASM Media Processing

Client-side audio/video pipelines combine WebCodecs for hardware
acceleration of video frame decoding/encoding with WASM for custom
processing (filters, analysis, transcoding fallback). wasm simd (128-bit)
vectorizes pixel and DSP loops for 2-4x gains; threads via
SharedArrayBuffer parallelize across cores. zero-copy transfer matters at
video rates: VideoFrame objects pass to workers without pixel copies, and
transferables move buffers; naive per-frame copies dominate budgets at
4K. real-time canvas or WebGL/WebGPU renders results; audio runs in
AudioWorklets with WASM DSP under 3ms quanta. Codec licensing and SAB
isolation headers are deployment constraints.

## Core Web Vitals Optimization

Google's field metrics: largest contentful paint (target < 2.5s) improves
via server response time, resource hints (preconnect, preload for hero
images and fonts), image optimization (AVIF/WebP, responsive srcset), and
prioritizing the critical path (inline critical CSS, defer non-critical
JS). interaction to next paint (target < 200ms, replaced FID) improves by
breaking long tasks (yield to main thread, scheduler.postTask), code
splitting, and web workers. cumulative layout shift (target < 0.1) is
prevented by reserving space: explicit dimensions on images/embeds,
font-display swap with size-adjusted fallbacks, and never inserting
content above existing content.
