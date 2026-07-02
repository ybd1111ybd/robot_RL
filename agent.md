# AI Harness Entry
This file is the default entry point for AI collaborators working on this repository.
Before any project work, print exactly:
`Harness loaded. Current phase: <phase>. Task domain: <domain>.`
Repository files are long-term memory. Chat history is not long-term memory.
Always use progressive disclosure: read only the files needed for the current task.
At the start of every project task, read these files first:
1. `agent.md`
2. `PROJECT_STATE.md`
3. `ROADMAP.md`
Then read only the relevant harness files:
- Task execution or implementation: `harness/tasks/`
- Interfaces, APIs, CLI contracts, data contracts: `harness/contracts/`
- Runtime, deployment, Docker, paths, environment setup: `harness/context/`
- Major technical choices and tradeoffs: `harness/decisions/`
Do not load all harness documents by default.
Do not treat previous chat messages as canonical project state.
If structure, interfaces, configuration, or behavior changes, update `PROJECT_STATE.md` and `CHANGELOG.md`.
If an interface changes, update the relevant file in `harness/contracts/`.
If deployment or runtime behavior changes, update the relevant file in `harness/context/`.
If a major technical decision is made, record it in `harness/decisions/`.
Keep changes scoped to the requested task and preserve existing user work.
Prefer existing project patterns over new abstractions.
Do not write business code when the task is only to update the harness.
Current phase is tracked in `PROJECT_STATE.md`.
Task domain must be inferred from the current request after reading the required files.
