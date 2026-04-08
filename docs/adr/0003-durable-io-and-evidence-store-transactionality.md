# ADR 0003: Durable IO and transactional evidence store

## Status
Accepted

## Decision
Critical artifacts use durable writes and the evidence store manifest is updated transactionally under lock.

## Rationale
`os.replace` without `fsync` is not sufficient for durable operational evidence.

## Consequences

- Critical artifacts survive crash windows more reliably.
- Evidence ingest is idempotent where intended and rejects checksum conflicts explicitly.
- Evidence expiration leaves tombstones instead of silent deletion.
