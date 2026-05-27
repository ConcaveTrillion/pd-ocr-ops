# CLAUDE — pdomain-ops

Python ops library for the `pd-*` suite: suite plumbing (registry, prefs,
sibling-spawn, desktop), GPU dispatch (StageDispatcher / LongJobRunner),
and schemas.emit. Imported by every `pd-*` SPA backend.

## Commands

| target | does |
|---|---|
| `make local-setup` | clone any missing sibling pd-* repos into the workspace |
| `make local-dev` | switch to local-dev mode (editable `../pdomain-book-tools` + marker) |
| `make local-check` | print local-dev mode + per-sibling resolution |
| `make local-upgrade-deps` | upgrade deps then restore editable siblings (local-mode only) |
| `make update-pd-deps` | bump pd-* sibling deps to registry latest; leaves diff for review |

See [workspace `docs/process/local-dev.md`](../docs/process/local-dev.md) for the canonical local-dev pattern (spec #362).
See [workspace `docs/process/update-pd-deps.md`](../docs/process/update-pd-deps.md) for the sibling-dep refresh pattern (spec #363).

## docs/ folder

This repo follows the workspace docs/ template — see [`docs/README.md`](docs/README.md). Active
folders: `architecture/`, `decisions/`, `plans/`, `process/`, `research/`,
`runbooks/`, `specs/`, `templates/`, `usage/`, plus parallel `archive/`
subfolders.

**Superpowers redirect.** When a superpowers skill (e.g. `brainstorming`,
`writing-plans`) instructs you to save to `docs/superpowers/specs/<file>.md`
or `docs/superpowers/plans/<file>.md`, save to `docs/specs/<file>.md` or
`docs/plans/<file>.md` instead. There is no `docs/superpowers/` subdirectory
in this repo.

<!-- workspace-process:start -->

## Before coding

These steps are workspace defaults for any coding task. **User-level settings
override them** — a user's own `~/.claude/CLAUDE.md`, `settings.json`, or a
direct instruction in the conversation takes precedence and may waive or
change any step below.

### Working principles

- **Use skills.** Invoke the relevant superpowers skill before starting —
  process skills first (`brainstorming`, `systematic-debugging`,
  `writing-plans`, `test-driven-development`), then implementation skills.
  If a skill applies, using it is not optional.
- **Delegate by default.** Dispatch subagents for non-trivial work: per-repo
  agents for repo changes, `Explore` for code searches. This keeps large tool
  output out of the parent context.
- **Parallelize.** Run independent tasks as concurrent subagents — multiple
  agent calls in a single message. Set `model: sonnet` on implementers and
  reviewers.

### Steps

1. **Check the working tree.** `git status --short`. Surface or resolve stray
   uncommitted work before starting — don't build on it.
2. **Read repo guidance.** This repo's `CLAUDE.md` and `CONVENTIONS.md` for
   repo-specific rules.
3. **Consult `docs/` for authoritative context** (whichever folders exist):
   `plans/` (the work plan), `specs/` (design specs — follow any `Spec:`
   pointer from the issue), `research/` (prior investigations), `decisions/`
   (ADRs / constraints), `architecture/` (shipped design).
4. **Check live issue status.** `gh issue view <N> --repo <owner/repo>` —
   confirm it isn't already closed; note its milestone.
5. **Check for in-flight work.** Open PRs and existing branches touching the
   same area, to avoid colliding with work-in-progress.
6. **Consult agent memory.** `.claude/agent-memory/<repo>/feedback_*.md` for
   corrections not yet promoted to `CONVENTIONS.md`.
7. **Locate code with `Explore` first.** Use an `Explore` subagent to find
   relevant files before broad `Read`/grep.
8. **Isolate in a worktree.** Never work directly in the interactive checkout
   at `/workspaces/ocr-container/<repo>/`. Use the `using-git-worktrees` skill
   to set up an isolated worktree. When delegating to a full-power
   implementation agent, pass `isolation: "worktree"` on the `Agent` call
   (skip for `-docs` agents and the `driver` agent). When an agent returns a
   worktree path + branch, use the `finishing-a-development-branch` skill to
   decide how to integrate.
9. **TDD.** Write the failing test first where the plan calls for it.
10. **Verify before committing.** Focused verification plus `make ci`.
11. **Commit locally; do not push** without explicit say-so.

<!-- workspace-process:end -->
