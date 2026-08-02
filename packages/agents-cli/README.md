# @declarative-ai/agents-cli

Delegated agents reached through a CLI subprocess - claude and codex - over the normalized AgentQuery seam shared with @declarative-ai/agents-api.

A workflow authored against one adapter runs against the other; only how the agent is reached — and therefore how its safety policy is enforced — differs.

```bash
npm i @declarative-ai/agents-cli
```

## Where it sits

Depends on `@declarative-ai/agents-api`, `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/ops`.
It has no dependencies outside the workspace.

Optional peer: `@modelcontextprotocol/sdk` — install only if you use the adapter that needs it.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
