# Cybersecurity & Cryptography: Technical Reference

## Post-Quantum Cryptography (NIST Standards)

Shor algorithm on a large fault-tolerant quantum computer breaks RSA and
elliptic-curve cryptography, motivating migration to quantum resistance.
NIST standardized lattice cryptography schemes built on the learning with
errors problem: ML-KEM (Kyber) for key encapsulation and ML-DSA (Dilithium)
for signatures, plus hash-based SLH-DSA (SPHINCS+) as a conservative
alternative. Costs are practical: ML-KEM-768 public key size is 1184 bytes
with 1088-byte ciphertexts — larger than X25519's 32 bytes but fast to
compute. Hybrid deployments (X25519+ML-KEM in TLS) hedge algorithm risk;
"harvest now, decrypt later" makes early migration urgent.

## Zero-Knowledge Proofs: zk-SNARKs vs zk-STARKs

zk-SNARKs (Groth16, PLONK) compile programs into an arithmetic circuit and
prove satisfiability with proofs under 200 bytes and millisecond
verification — extreme succinctness — using pairings over elliptic curves.
Groth16 needs a per-circuit trusted setup; universal-setup systems (PLONK,
Marlin) use one ceremony via a polynomial commitment scheme (KZG).
zk-STARKs remove trusted setup entirely, relying on hash-based FRI
commitments: transparent and plausibly post-quantum, but proofs are tens
to hundreds of kilobytes. Choose SNARKs for on-chain verification cost,
STARKs for transparency, PQ hedging, and large-batch proving throughput.

## OAuth 2.1 / OIDC with PKCE

OAuth 2.1 consolidates best practice: authorization code flow with PKCE is
mandatory for public clients and the implicit grant is removed. The client
generates a random code verifier and sends its SHA-256 hash as the code
challenge; at token exchange it presents the verifier, so an intercepted
authorization code alone is useless — blocking code injection and replay
attack vectors. Confidential clients still authenticate with a client
secret or private-key JWT. OpenID Connect layers identity on top: the ID
token's nonce binds it to the session, and at_hash binds the access token.
Refresh tokens for public clients must rotate with reuse detection.

## Rust Memory Safety

Rust enforces memory safety at compile time through ownership: each value
has one owner, moves transfer it, and the borrow checker admits either one
mutable or many shared references — never both — eliminating use-after-
free, double-free, and iterator invalidation without garbage collection.
lifetime analysis guarantees references never outlive referents. Safe Rust
also prevents data race bugs: Send/Sync traits gate cross-thread sharing.
These guarantees are zero-cost abstraction — compiled code matches manual
C performance. Unsafe blocks permit raw-pointer operations for FFI and
data structures; soundness then depends on encapsulating invariants,
since violating them reintroduces undefined behavior.

## WebAssembly Sandboxing

WASM modules execute against linear memory — a contiguous, growable buffer
fully isolated from host memory; every access is subject to bounds
checking, often hardware-assisted via guard pages. Control flow integrity
is structural: typed function tables and validated jump targets prevent
ROP-style attacks. Imports follow capability-based security — a module
touches only host functions explicitly granted (WASI). Versus native
process isolation: cheaper instantiation and in-process embedding, but
shared-process side channels require spectre mitigation (site isolation,
bounded speculation). Runtimes add defense-in-depth: Wasmtime pools,
per-instance limits, and fuel metering.

## TLS 1.3 Handshake

TLS 1.3 cuts handshake round trips to one RTT: ClientHello carries
diffie-hellman ephemeral key shares, and the server's first flight
completes key establishment — versus two RTTs in TLS 1.2. 0-rtt resumption
sends early data under a PSK from a prior session, saving the last RTT at
the cost of replayability (restricted to idempotent requests, single-use
tickets). Legacy cipher suites are gone: only AEAD suites (AES-GCM,
ChaCha20-Poly1305) with forward secrecy remain, and the handshake itself
is encrypted after the key share. Downgrade protection embeds a sentinel
in the server random, detecting version-rollback attempts.

## Software Supply Chain Security (SBOM/SLSA)

A software bill of materials enumerates all components; spdx and cyclonedx
are the interchange formats, enabling continuous dependency vulnerability
matching against advisories (OSV, NVD) and license audit. SLSA levels
harden the build pipeline: provenance attests what source and build system
produced an artifact, and attestation is signed (Sigstore cosign, in-toto)
so consumers verify integrity before deployment. Reproducible builds make
tampering detectable by rebuild comparison. Controls target real attacks:
dependency confusion, typosquatting, compromised build servers
(SolarWinds), and malicious maintainer takeovers (xz backdoor).

## Fuzzing Methodologies

Coverage-guided greybox fuzzers (AFL++, libFuzzer) instrument the target
so code coverage feedback steers corpus mutation: inputs reaching new
edges join the corpus, evolving deep test cases. Instrumentation includes
sanitizers — ASan (heap overflows, UAF), MSan, UBSan — converting silent
corruption into crashes. crash deduplication groups findings by stack
hash for triage. Structured formats need grammar-aware or mutation
dictionaries; protocol targets use snapshot fuzzing. OSS-Fuzz runs this
continuously at scale, demonstrating that persistent fuzzing plus
sanitizers finds memory-safety bugs static analysis misses.

## HSMs and Trusted Execution Environments

Hardware security modules provide tamper-resistant key protection: keys
generate and operate inside FIPS 140-3 validated boundaries and never
leave in plaintext — the root of trust for CAs, payments, and KMS
backends. TEEs isolate computation on general CPUs: intel sgx enclaves
encrypt memory and offer remote attestation verification proving genuine
enclave identity before secret provisioning; arm trustzone splits secure
and normal worlds for mobile keystores. TEE caveats: side-channel leakage
(cache timing, speculative execution) has repeatedly breached enclave
confidentiality, so defense assumes microarchitectural attacks and
combines TEEs with algorithmic hardening.

## Kerberos Attacks in Active Directory

Kerberos issues a tgt ticket from the KDC, encrypted with the krbtgt hash;
service tickets carry a PAC listing group memberships. Attack paths:
Kerberoasting requests tickets for any service principal name and cracks
service-account passwords offline; pass the ticket replays stolen tickets
from LSASS; overpass-the-hash converts an NTLM hash into a TGT. A stolen
krbtgt hash enables golden tickets — forged TGTs with arbitrary
privileges — while skipped pac validation lets forged PACs escalate.
Defenses: managed service accounts with long random passwords, AES-only
Kerberos, protected users group, credential guard, and monitoring for
encryption-type anomalies.
