# Roadmap

Future work that is not required for the current 1.0 support matrix. Current readiness and tier decisions live in [docs/1.0-READINESS.md](docs/1.0-READINESS.md).

## Research And Model Quality

- Promote CLAP meeting-room and software-demo fixtures once they separate cleanly enough to become validated calibration inputs.
- Continue measuring whether CLAP-derived audio context improves summaries enough to justify the extra prompt and latency cost.
- Improve retrieval routing across transcript, slides, vision, and summary indexes with reranking tuned for long-video Q&A.
- Add incremental recomputation with step-level caching, lineage, and versioning for transcripts, frames, vision metadata, summaries, and embeddings.

## Runtime And Hardware

- Promote ROCm vLLM only after Strix Halo startup, memory contention, and teardown behavior are reliable enough for default use.
- Reduce AMD cold-start overhead for large embedding models so batch tails are not dominated by first-load startup costs.
- Extend the runtime residency planner beyond current free-memory placement toward more dynamic isolation, sharing, and multi-GPU splitting decisions.
- Standardize and validate the Strix Halo host baseline around known-good ROCm and kernel combinations.

## Product Workflows

- Add optional live transcription with audio capture or mixing for real-time meetings and demos.
- Add external sync/export targets for action items once the integration format is chosen.
- Improve log and UI clarity for alternate ASR backends such as Canary and Granite.

## Security And Administration

- Add per-user auth, RBAC, and stronger session/device management if EchOnyx moves beyond the current trusted single-admin model.
- Add MFA only if deployment needs expand beyond trusted local-admin use.
- Add stricter export/search quotas and stronger reverse-proxy headers such as CSP once production traffic and ingress patterns are known.
- Review artifact-retention controls for uploads, transcripts, slides, embeddings, exports, and audit logs.
