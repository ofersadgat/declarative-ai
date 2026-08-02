# @declarative-ai/agents-api

Delegated agents reached through an in-process SDK, over the normalized AgentQuery seam shared with @declarative-ai/agents-cli.

Its entry declares `policyEnforcement: "callback"` — it routes the agent's tool approvals back through `ctx.approve`.

```bash
npm i @declarative-ai/agents-api
```

## Where it sits

Depends on `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/ops`, `@declarative-ai/permissions`.
It has no dependencies outside the workspace.

Optional peer: `@anthropic-ai/claude-agent-sdk` — install only if you use the adapter that needs it.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
