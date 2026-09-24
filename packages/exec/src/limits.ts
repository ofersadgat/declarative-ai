/**
 * The limits board — ONE place, per host process, that knows how much of each account's allowance is
 * spent.
 *
 * A limit is global to the account behind a route: not to a session, a task or a project, and not to
 * an executor either — the Agent SDK route and the `claude` CLI route share one subscription, and a
 * meter must have somewhere to ask with no executor running at all. So the board is keyed by ACCOUNT
 * and handed to every executor's session layer (`withSessionPosition`), which reports what comes back
 * on the stream and stamps what goes out.
 *
 * Two rules decide when to ask rather than wait, and they answer different failures:
 *
 * 1. SENT SINCE HEARD. Messages have been going out for longer than `refreshAfterSendingMs` with no
 *    update coming back — the account is being spent and the number is not moving, which is what a
 *    provider that reports limits only "when they change" produces. Checked on every `sent`.
 * 2. TOO OLD, WHATEVER HAPPENED. A reading older than `refreshAfterMs`, or one whose earliest reset has
 *    passed (it describes a window that no longer exists). Checked by a timer that runs ONLY while
 *    something subscribes — a meter on screen — and on demand by {@link LimitsBoard.fresh}. With
 *    nobody listening and nothing asking, nothing is fetched.
 *
 * A refresh is something an executor OFFERS for an account; an account may have none, and then its
 * reading simply ages. A refresh that fails is not an error anyone sees: the state stays as it was.
 */
import { mergeLimitReadings, type LimitReading } from "@declarative-ai/ops";

/** Which account a reading belongs to — provider and account where known, else the route. */
export type AccountKey = string;

/** What the board knows about one account. */
export interface LimitState {
  /** The latest reading from any session on any executor of this account; `null` before the first. */
  reading: LimitReading | null;
  /** When that reading arrived (ISO-8601). */
  updatedAt: string | null;
  /** When any executor of this account last sent a message (ISO-8601). */
  lastSentAt: string | null;
  /** A refresh is in flight; a second is never started beside it. */
  refreshing: boolean;
}

/** A refresh an executor offers: fetch the account's windows without spending a turn. */
export type LimitsRefresh = (signal?: AbortSignal) => Promise<LimitReading | undefined>;

export interface Disposable {
  dispose(): void;
}

export interface LimitsBoard {
  /** A reading arrived from a session. Latest wins, merged with what was known. */
  report(account: AccountKey, reading: LimitReading): void;
  /** Something is about to be sent on this account (rule 1's stamp). */
  sent(account: AccountKey): void;
  /** Register the way to refresh an account. The last offer wins; disposing removes it. */
  offerRefresh(account: AccountKey, refresh: LimitsRefresh): Disposable;
  /** Never throws; `reading` may be null. */
  state(account: AccountKey): LimitState;
  /** Every account known, whether or not any executor is running. */
  all(): ReadonlyMap<AccountKey, LimitState>;
  /**
   * Called on every change. A WATCHING subscriber (the default — a meter on screen) keeps rule 2's
   * timer running: the first starts it, the last to leave stops it. A host that only wants to hear
   * about changes — to persist them, to forward them — subscribes with `watch: false` and starts
   * nothing.
   */
  subscribe(listener: (account: AccountKey, state: LimitState) => void, options?: { watch?: boolean }): Disposable;
  /** Refresh now if a refresh is offered (always, or only when older than `olderThanMs`) — what a person pressing Refresh asks for. */
  refresh(account: AccountKey, options?: { olderThanMs?: number }): Promise<LimitState>;
  /** Rule 2 on demand: the state, refreshed first when it is too old. For a reader that must decide now. */
  fresh(account: AccountKey): Promise<LimitState>;
  /** Stop timers and drop listeners. */
  close(): void;
}

export interface LimitsBoardOptions {
  /** Rule 2: a reading older than this is refreshed while someone is watching. Default 30 minutes. */
  refreshAfterMs?: number;
  /** Rule 1: sending for this long with nothing heard triggers a refresh. Default 5 minutes. */
  refreshAfterSendingMs?: number;
  /** How often rule 2's timer looks. Default 60 seconds. */
  tickMs?: number;
  /** What the host remembered from before (the board is persisted by the host, not here). */
  initial?: Iterable<readonly [AccountKey, LimitState]>;
  /** Injectable clock for tests. */
  now?: () => number;
  /** How long one refresh may take before it is abandoned. Default 20 seconds. */
  refreshTimeoutMs?: number;
}

/**
 * What a transport tells the session layer about allowance (`ExecServices.usage`). Keyed by ROUTE —
 * the transport knows which route it is and nothing about accounts; the session layer maps one to
 * the other.
 */
export interface UsageReporter {
  /** A call is about to be sent on this route (rule 1's stamp). */
  sent(route: string): void;
  /** A limit reading came back. */
  limits(reading: LimitReading): void;
}

/** The reporter a session layer hands down: routes mapped to accounts, readings to the board. */
export function boardReporter(board: LimitsBoard, accountOf: (route: string) => AccountKey | undefined): UsageReporter {
  return {
    sent(route) {
      const account = accountOf(route);
      if (account !== undefined) board.sent(account);
    },
    limits(reading) {
      const account = accountOf(reading.route);
      if (account !== undefined) board.report(account, reading);
    },
  };
}

const EMPTY: LimitState = Object.freeze({ reading: null, updatedAt: null, lastSentAt: null, refreshing: false });

/** Build a board. It holds no I/O of its own: refreshes are offered to it, persistence is the host's. */
export function createLimitsBoard(options: LimitsBoardOptions = {}): LimitsBoard {
  const refreshAfterMs = options.refreshAfterMs ?? 30 * 60_000;
  const refreshAfterSendingMs = options.refreshAfterSendingMs ?? 5 * 60_000;
  const tickMs = options.tickMs ?? 60_000;
  const timeoutMs = options.refreshTimeoutMs ?? 20_000;
  const now = options.now ?? Date.now;
  const states = new Map<AccountKey, LimitState>();
  for (const [k, s] of options.initial ?? []) states.set(k, { ...s, refreshing: false });
  const refreshers = new Map<AccountKey, LimitsRefresh>();
  // When the first message went out with nothing heard since — rule 1's clock.
  const sendingSince = new Map<AccountKey, number>();
  const inflight = new Map<AccountKey, Promise<LimitState>>();
  const listeners = new Set<(account: AccountKey, state: LimitState) => void>();
  // When each account was last asked — so an account whose refresh learns nothing (an older binary,
  // a signed-out login) is not asked again on every tick. The rules wait `refreshAfterMs` between
  // attempts; a person pressing Refresh does not.
  const triedAt = new Map<AccountKey, number>();
  let watchers = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  let closed = false;

  const iso = (): string => new Date(now()).toISOString();
  const put = (account: AccountKey, next: LimitState): void => {
    states.set(account, next);
    for (const l of [...listeners]) {
      try {
        l(account, next);
      } catch {
        // A listener's failure is its own; the board keeps going.
      }
    }
  };
  const stateOf = (account: AccountKey): LimitState => states.get(account) ?? EMPTY;

  /** Rule 2's test: no reading, too old, or a reset that has passed. */
  const stale = (s: LimitState, olderThanMs: number): boolean => {
    if (s.reading === null || s.updatedAt === null) return true;
    const t = now();
    if (t - Date.parse(s.updatedAt) > olderThanMs) return true;
    for (const w of s.reading.windows) {
      if (w.resetsAt !== null && Date.parse(w.resetsAt) <= t) return true;
    }
    return false;
  };

  const report = (account: AccountKey, reading: LimitReading): void => {
    if (closed) return;
    const s = stateOf(account);
    sendingSince.delete(account);
    put(account, { ...s, reading: mergeLimitReadings(s.reading, reading), updatedAt: iso() });
  };

  /** Asked recently enough that the rules should not ask again yet. */
  const triedRecently = (account: AccountKey): boolean => {
    const t = triedAt.get(account);
    return t !== undefined && now() - t < refreshAfterMs;
  };

  const runRefresh = (account: AccountKey): Promise<LimitState> => {
    const going = inflight.get(account);
    if (going !== undefined) return going;
    const refresh = refreshers.get(account);
    if (refresh === undefined || closed) return Promise.resolve(stateOf(account));
    triedAt.set(account, now());
    put(account, { ...stateOf(account), refreshing: true });
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), timeoutMs);
    const p = (async (): Promise<LimitState> => {
      try {
        const reading = await Promise.race([
          refresh(abort.signal),
          new Promise<undefined>((resolve) => abort.signal.addEventListener("abort", () => resolve(undefined), { once: true })),
        ]);
        if (reading !== undefined) report(account, reading);
      } catch {
        // Not an error anyone sees: the state simply stays old, and its age shows.
      } finally {
        clearTimeout(timeout);
        inflight.delete(account);
        put(account, { ...stateOf(account), refreshing: false });
      }
      return stateOf(account);
    })();
    inflight.set(account, p);
    return p;
  };

  const tick = (): void => {
    for (const account of new Set([...refreshers.keys()])) {
      if (stale(stateOf(account), refreshAfterMs) && !triedRecently(account)) void runRefresh(account);
    }
  };

  return {
    report,
    sent(account) {
      if (closed) return;
      const t = now();
      put(account, { ...stateOf(account), lastSentAt: new Date(t).toISOString() });
      const since = sendingSince.get(account);
      if (since === undefined) {
        sendingSince.set(account, t);
        return;
      }
      if (t - since > refreshAfterSendingMs) {
        sendingSince.set(account, t);
        if (!triedRecently(account) || t - (triedAt.get(account) ?? 0) > refreshAfterSendingMs) void runRefresh(account);
      }
    },
    offerRefresh(account, refresh) {
      refreshers.set(account, refresh);
      // A watcher already on screen gets a first reading for an account it had none for.
      if (timer !== undefined && stale(stateOf(account), refreshAfterMs) && !triedRecently(account)) void runRefresh(account);
      return {
        dispose() {
          if (refreshers.get(account) === refresh) refreshers.delete(account);
        },
      };
    },
    state: stateOf,
    all: () => new Map(states),
    subscribe(listener, opts) {
      listeners.add(listener);
      const watching = opts?.watch !== false;
      if (watching) {
        watchers += 1;
        if (timer === undefined && !closed) {
          timer = setInterval(tick, tickMs);
          (timer as { unref?: () => void }).unref?.();
          tick();
        }
      }
      let disposed = false;
      return {
        dispose() {
          if (disposed) return;
          disposed = true;
          listeners.delete(listener);
          if (!watching) return;
          watchers -= 1;
          if (watchers === 0 && timer !== undefined) {
            clearInterval(timer);
            timer = undefined;
          }
        },
      };
    },
    refresh(account, opts) {
      if (opts?.olderThanMs !== undefined && !stale(stateOf(account), opts.olderThanMs)) return Promise.resolve(stateOf(account));
      return runRefresh(account);
    },
    fresh(account) {
      return stale(stateOf(account), refreshAfterMs) && !triedRecently(account) ? runRefresh(account) : Promise.resolve(stateOf(account));
    },
    close() {
      closed = true;
      if (timer !== undefined) clearInterval(timer);
      timer = undefined;
      listeners.clear();
    },
  };
}
