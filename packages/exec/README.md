# @declarative-ai/exec

The one execution seam - Executor.start(op, ctx) - with handles, composition, memoization, AIMD rate limiting, deadlines, retry, and append-only sessions.

One seam, `Executor.start(op, ctx)`. It knows nothing about LLMs, validation, permissions, or filesystems; those declare their own seams by augmenting `ExecServices`.

```bash
npm i @declarative-ai/exec
```

## Where it sits

Depends on `@declarative-ai/json`, `@declarative-ai/ops`.
It has no dependencies outside the workspace.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
