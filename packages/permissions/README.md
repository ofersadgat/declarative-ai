# @declarative-ai/permissions

The tool-call permission model for declarative-ai: ExecPolicy, Approver, profile by mode resolution, and baselines.

Its two axes are orthogonal, and it declares its own seams on `ExecServices`, so `exec` does not know permissions exist.

```bash
npm i @declarative-ai/permissions
```

## Where it sits

Depends on `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/ops`.
It has no dependencies outside the workspace.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
