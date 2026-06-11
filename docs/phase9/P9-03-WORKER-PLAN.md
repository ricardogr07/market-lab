# P9-03 Worker Plan: Frozen BTC Shadow Contract

## Packet Identity

- Branch: `feature/phase-9-btc-shadow-lock`
- Pull request: `feat(phase9): freeze BTC shadow candidate contract`
- Dependency: P9-01 and the merged P9-02 documentation baseline
- Historical candidate code lock: `ce01124`

P9-03 freezes the retained Phase 8 BTC candidate as a signals-only Phase 9
shadow contract. It supplies the configuration and fail-closed verifier that
P9-04 must call before producing a decision.

P9-03 does not fetch market data, run models, schedule work, create decision
records, write reports, deploy Azure resources, request approval, or call a
broker. Those behaviors remain in later packets.

## Frozen Protocol

| Field | Approved value |
| --- | --- |
| Candidate ID | `btc-phase9-shadow-v1` |
| Behavior version | `btc-phase8-guarded-gate-v1` |
| Protocol start | `2026-06-03` |
| Protocol end | `2027-06-02` |
| Earliest final labeled evaluation | `2027-06-16` |
| Target maturity lag | `14` daily bars |
| Symbol and interval | `BTC-USD`, `1d` |
| Artifact root | `artifacts/phase9-shadow` |
| Paper execution | Disabled |

The checked-in config is
`configs/experiment.btc_phase9_shadow_daily.yaml`. Its strategy-affecting
settings must remain semantically identical to
`configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml`.

The Phase 9 config ends at the protocol boundary. P9-04 must pass a runtime
completed-bar as-of cutoff and must never modify or generate a replacement for
the checked-in config. Missed protocol dates must be recorded explicitly by
P9-04 and cannot be silently reconstructed with data that was unavailable at
the original decision time.

## Hash Contract

The verifier uses SHA-256 over canonical UTF-8 JSON with sorted keys and compact
separators.

The full config hash covers the complete YAML mapping except the two declared
hash values. It therefore detects changes to behavior, protocol metadata,
safety settings, paths, and artifact options. The approved full config hash is:

```text
d439acca79ca2108a4d907452b5d442ab67b319d430440d07f14f9adc1295f18
```

The behavior hash covers the behavior version plus normalized values from:

- data symbols, research start date, and interval
- all feature, target, portfolio, baseline, model, and evaluation settings

It excludes the experiment name, data end date, cache and artifact paths, and
shadow protocol metadata. The approved behavior hash is:

```text
71beba28529abba3482145094654c5eaf8f12355d92a93830fe746a241129550
```

The expected hashes are independently pinned in
`marketlab.shadow.contract`. Hash declarations in the YAML cannot approve a
changed candidate because the verifier requires the declarations, calculated
digests, and independent registry to agree.

## Public Contract

P9-03 adds:

```python
from marketlab.shadow import verify_shadow_contract

contract = verify_shadow_contract(
    "configs/experiment.btc_phase9_shadow_daily.yaml"
)
```

Successful verification returns `VerifiedShadowContract`, including the typed
experiment config, candidate identity, protocol dates, maturity lag, code lock,
artifact root, and both calculated hashes. Any mismatch raises
`ShadowContractError` before downstream shadow work can begin.

The verifier rejects:

- unknown candidate IDs or changed registry-controlled metadata
- loading the candidate outside `configs/experiment.btc_phase9_shadow_daily.yaml`
- malformed or reordered protocol dates
- any full-config or behavior drift
- a non-BTC symbol, non-daily interval, or non-14-bar target
- an artifact root that differs from the approved shadow root
- `paper.enabled: true`

## Implementation Order

1. Add the typed optional `shadow` configuration section without changing
   existing config defaults.
2. Add the frozen Phase 9 YAML as an operational mirror of the retained Phase 8
   behavior.
3. Add canonical hashing, the independent approval registry, and the verifier.
4. Add mutation, mirror, safety, documentation, lint, and type-check coverage.

The worker owns the config schema, YAML, verifier, tests, and documentation. QA
must verify deterministic hashing and fail-closed mutations. The critic must
review the hash boundary and prevent P9-04 decision or journal behavior from
entering this packet. The financial reviewer must confirm that the behavior
payload remains an exact Phase 8 strategy mirror.

## Validation And Acceptance

Run:

```text
python -m pytest -q tests/unit/test_phase9_shadow_contract.py tests/unit/test_phase9_plan_docs.py tests/unit/test_config.py
python -m ruff check src/marketlab/config.py src/marketlab/shadow tests/unit/test_phase9_shadow_contract.py tests/unit/test_phase9_plan_docs.py
python -m mypy src/marketlab/config.py src/marketlab/shadow/contract.py
python -m mkdocs build --strict
py -3.14 -m tox -e preflight
git diff --check
```

P9-03 is complete only when the checked-in Phase 9 config verifies against the
independent registry, its behavior hash equals the retained Phase 8 behavior
hash, mutations fail closed, paper remains disabled, and P9-04 can consume the
typed verified contract without introducing another lock mechanism.
