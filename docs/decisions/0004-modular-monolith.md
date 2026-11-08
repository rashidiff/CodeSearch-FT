# ADR 0004: Modular monolith

- **Status:** Accepted by approved project design
- **Context:** Indexing, retrieval, training, and API operations share local artifacts and are developed by one project without a distributed deployment need.
- **Decision:** Use one installable `src/` Python package with domain contracts, application services, and focused adapters; CLI and API compose those services.
- **Alternatives:** A notebook-only design cannot serve as the product; microservices add deployment/operations overhead; a single flat module weakens meaningful testing and replacement boundaries.
- **Consequences:** Module dependencies remain explicit and acyclic; test doubles are used only at defined ports. Phase 1 establishes package boundaries and a minimal CLI/API without implementing later-phase product services.
