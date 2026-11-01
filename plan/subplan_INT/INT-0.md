# INT-0 — Integration of the Codex lane and the M4 line

Parent step: INT-0 in [the shared master](../PLAN.md#12-authoritative-execution-order).
State: CLOSED_INTEGRATED. Recorded 2026-11-01.

## Objective and scope

Bring the two lines of cuda4AS work that never met into one history without
changing the content of either: the Codex lane on the Linux box (26 commits on
`codex/*` branches from `ad30bbd`, plus the uncommitted canonical plan tree)
and the M4 Mac mini line on `origin/master` (three commits, 2026-09-20 to
2026-10-26, written by an Antigravity session and the M4 "cuda4AS Committee").

This is the unique retrospective subplan for the named closed step. Its child
IDs identify the historical work breakdown; they do not claim new executions
or reopen the step. No executor assignment follows from this document.

## Inputs and evidence

- Handover (Linux box): `~/.claude/handover/2026-11-01-cuda4as-mac-integration.md`,
  from the session "cuda4AS Committee" to the session "cuda4AS executor".
- Report: `~/.claude/handover/2026-11-01-cuda4as-mac-integration.REPORT.md`.
- Repository state after the step: `master` =
  `382a66f1be21c8bbc1212052a249e39cd61b40d3`, parents `1f75183` (M4 line)
  and `be172c09aa3f2708f50282bbe77a3ec9871efdeb` (the Codex lane as found,
  committed on `codex/m2a-r1-clang-semantic-import` and tagged
  `pre-mac-integration-2026-11-01`). Branch, tag and `master` are on `origin`,
  verified by `git ls-remote`.

## Execution order — historical, non-executable

| Child ID | Completed scope | Evidence or decision |
|---|---|---|
| INT-0-1 | Snapshot the as-found Codex-era working tree as one commit: `plan/` (26 files), `handover/` (3), `RESULTS/m2a/` (2), eight planning-authority banners, one `.gitignore` line; tag; push; verify. | `be172c0`; tag object `b24f63a` dereferencing to `be172c0` on `origin`. The takeover rule (snapshot first) applied in both directions: Codex locally, Antigravity remotely. |
| INT-0-2 | Fast-forward local `master` to `origin/master`. | `1f75183`; local `master` had no commits of its own. |
| INT-0-3 | Merge the Codex branch into `master` with a merge commit and resolve the single conflict. | `.gitignore` only: the union of both blocks plus `/cuda-examples/` (a 978 MB third-party tree with no repository of its own, never tracked). `README.md` merged automatically: Codex banner on top, M4 body below. |
| INT-0-4 | Sanity checks. | Every path of both parents present in the result; no conflict markers; clean tree; six local `codex/*` branches unmoved; two minor deviations recorded (one trailing newline, a sixth branch at the old base). |
| INT-0-5 | Push `master` on the user's yes; verify. | `origin/master` = `382a66f`. |
| INT-0-6 | Inventory of the M4 documents for INT-1, as facts with `file:line`. | Report §4, items 1–8. |

## Acceptance and classification

Accepted by the Committee on 2026-11-01 after independent verification of the
merge parents, the remote refs, the clean tree and the two-file modification
set against the M4 line. The step changed no content of either line beyond
`.gitignore`, accepted no evidence and made no compatibility claim.

## Preserved limits and downstream use

The parent YFCE repository was read only: its `cuda4AS` gitlink is `ad30bbd`,
its local `master` is nine commits ahead of the last-fetched `origin/master`
and its remote `master` has moved; that integration is a separate handover
(master §16). The merge left the README's banner and the M4 text side by side
with their contradictions; INT-1 settled authority, and the README pointers
are corrected by the INT-1 executor handover.

Current execution eligibility comes only from the master and an authorized
handover. New experiments append new immutable evidence; they never replace
this step's record.
