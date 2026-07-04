# fastapi-disconnect

Cancel FastAPI request handlers when the client disconnects.

By default, when a client disconnects (timeout, closed tab, network failure),
FastAPI keeps executing the request handler to completion. For long-running
handlers — an AI flow making paid LLM calls, a heavy database query — that
means paying for work nobody is waiting for. This library cancels the handler
the moment the client goes away, by listening for the ASGI `http.disconnect`
event (no polling of `request.is_disconnected()`).

## Installation

```sh
pip install fastapi-disconnect
```

## Usage

App-wide, via middleware — protects every route, no handler changes:

```python
from fastapi import FastAPI
from fastapi_disconnect import CancelOnDisconnectMiddleware

app = FastAPI()
app.add_middleware(
    CancelOnDisconnectMiddleware,
    exclude_paths=[r"/uploads/.*"],   # optional; re.fullmatch against the path
    on_disconnect=log_cancellation,   # optional; sync or async, gets the ASGI scope
)
```

`on_disconnect` runs only when a *premature* disconnect cancelled a request
(not on normal connection close) — a natural place for a metric counting the
work cancellation saved you.

Per-endpoint, via decorator — applied below the route decorator:

```python
from fastapi_disconnect import cancel_on_disconnect

@app.get("/generate")
@cancel_on_disconnect
async def generate() -> str:
    return await expensive_llm_flow()
```

Cooperative, via dependency — for handlers that want to decide *when* to
stop instead of being cancelled preemptively (e.g. finish the current
step, checkpoint, then exit):

```python
from fastapi_disconnect import Disconnected

@app.get("/agent")
async def agent(disconnected: Disconnected) -> Result:
    for step in plan:
        if disconnected.is_set():        # free local check, no polling
            return partial_result
        await run_step(step)
```

`Disconnected` is an annotated alias for
`Annotated[asyncio.Event, Depends(disconnected_event)]`; the event is set the
moment the client disconnects and can also be awaited (`disconnected.wait()`).
Use it *instead of* the decorator/middleware on a route — combined with them,
preemptive cancellation fires first (though the event is still set, which is
handy inside shielded cleanup). All three mechanisms share a single
per-request watcher, so any combination is safe.

On disconnect, the handler receives a regular `asyncio.CancelledError` at its
next `await`, so `finally` blocks and async context managers run as usual.
The decorator produces a `499 Client Closed Request` response (which goes
nowhere — the client is gone — but keeps logs meaningful); the middleware
simply stops.

## Cleanup on cancellation

Catching `CancelledError` for cleanup is supported, with one rule: the
cancellation is **level-triggered** (anyio cancel-scope semantics). Once the
client is gone, every subsequent `await` inside the handler raises
`CancelledError` again — so asynchronous cleanup must run in a shielded scope:

```python
import anyio

@app.get("/generate")
@cancel_on_disconnect
async def generate() -> str:
    try:
        return await expensive_llm_flow()
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            await release_resources()  # runs to completion
        raise
```

An unshielded `await` in an `except CancelledError` or `finally` block is
re-cancelled at its first checkpoint. (`asyncio.shield()` is not a substitute:
it detaches the inner call, but the awaiting handler is still re-cancelled, so
the cleanup continues without you.) Synchronous cleanup needs no shielding.
Re-raise after cleaning up — a handler that swallows the cancellation and
returns a value is treated as having completed normally.

## License

MIT
