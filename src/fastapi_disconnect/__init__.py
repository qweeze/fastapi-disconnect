import asyncio
import contextlib
import functools
import inspect
import math
import re
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from typing import Annotated, Any

import anyio
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

try:  # fastapi is only needed for the `Disconnected` convenience alias
    import fastapi
except ModuleNotFoundError:  # pragma: no cover
    fastapi = None  # type: ignore[assignment]

__all__ = [
    "CancelOnDisconnectMiddleware",
    "Disconnected",
    "OnDisconnect",
    "cancel_on_disconnect",
    "disconnected_event",
]

type OnDisconnect = Callable[[Scope], Awaitable[None] | None]
"""Callback invoked after a premature disconnect cancelled a request; sync or async."""

HTTP_499_CLIENT_CLOSED_REQUEST = 499

_REQUEST_PARAM = "__cancel_on_disconnect_request"
_SCOPE_KEY = "fastapi_disconnect.event"


@contextlib.asynccontextmanager
async def _disconnect_event(scope: Scope, receive: Receive) -> AsyncIterator[asyncio.Event]:
    """Get the shared per-request event that is set on client disconnect.

    The first consumer creates the event, stores it in the ASGI scope and
    starts the watcher task; later consumers reuse it. This is the only
    receive-reading loop in the library — one reader per channel, so the
    mechanisms compose on any ASGI server.

    The watcher consumes the messages it reads, so while it runs nothing
    else may read the same channel (FastAPI-parsed body params are fine:
    they are consumed before dependencies and the handler run). A caller
    that needs the messages passes a teeing `receive`, as
    `CancelOnDisconnectMiddleware` does.
    """
    existing: asyncio.Event | None = scope.get(_SCOPE_KEY)
    if existing is not None:
        yield existing
        return

    event = asyncio.Event()
    scope[_SCOPE_KEY] = event

    async def watch() -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                event.set()
                return

    watcher = asyncio.create_task(watch())
    try:
        yield event
    finally:
        watcher.cancel()


@contextlib.asynccontextmanager
async def _scope_cancelled_when(
    trigger: Callable[[], Coroutine[Any, Any, object]],
) -> AsyncIterator[anyio.CancelScope]:
    """Run the block in a cancel scope, cancelled when `trigger` completes.

    A bare `CancelScope` (not a task group) on purpose: cancellation can only
    land inside the block, and exceptions leaving the block are not wrapped
    into `ExceptionGroup` (anyio 4 task groups would wrap them, breaking
    FastAPI's `HTTPException` handling).
    """
    with anyio.CancelScope() as cancel_scope:

        async def watch() -> None:
            await trigger()
            cancel_scope.cancel()

        watcher = asyncio.create_task(watch())
        try:
            yield cancel_scope
        finally:
            watcher.cancel()


def cancel_on_disconnect[**P, R](
    handler: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R | Response]]:
    """Cancel the request handler as soon as the client disconnects.

    The handler does not need to declare a `Request` parameter. The wrapper
    advertises one extra keyword-only `Request` parameter via
    `__signature__`; FastAPI injects it and the wrapper strips it off before
    calling the handler. `Request`-annotated params never appear in OpenAPI.
    Called directly (e.g. in unit tests), the handler runs unguarded.
    Apply below the route decorator: `@app.get(...)` first, this second.
    """
    if not inspect.iscoroutinefunction(handler):
        raise TypeError(
            f"@cancel_on_disconnect requires an async def handler, got {handler!r}: "
            "sync handlers run in a threadpool and cannot be cancelled"
        )

    @functools.wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | Response:
        request = kwargs.pop(_REQUEST_PARAM, None)
        if request is None:  # direct call, not via FastAPI
            return await handler(*args, **kwargs)

        assert isinstance(request, Request)
        async with (
            _disconnect_event(request.scope, request.receive) as event,
            _scope_cancelled_when(event.wait) as guard,
        ):
            result: R | Response = await handler(*args, **kwargs)

        if guard.cancelled_caught:
            # Nobody is listening anymore; the response goes nowhere.
            result = Response(status_code=HTTP_499_CLIENT_CLOSED_REQUEST)

        elif isinstance(result, StreamingResponse):
            warnings.warn(
                f"{handler.__qualname__} returned a StreamingResponse: the stream"
                " body runs after the cancel-on-disconnect guard has exited and"
                " will not be cancelled; use CancelOnDisconnectMiddleware to"
                " cover streaming responses",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    # eval_str resolves stringified annotations with the handler's globals,
    # which FastAPI could not do itself through a wrapper from another module.
    signature = inspect.signature(handler, eval_str=True)
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                _REQUEST_PARAM, kind=inspect.Parameter.KEYWORD_ONLY, annotation=Request
            ),
        ]
    )
    return wrapper


async def disconnected_event(request: Request) -> AsyncIterator[asyncio.Event]:
    """FastAPI dependency yielding an event that is set on client disconnect.

    For cooperative handling — deciding yourself when to stop, instead of
    being cancelled preemptively by the decorator or middleware:

        @app.get("/agent")
        async def agent(disconnected: Disconnected) -> str:
            for step in plan:
                if disconnected.is_set():
                    return partial_result
                await run(step)

    Use it *instead of* the decorator/middleware on a route: combined with
    them, preemptive cancellation fires first (the event is still set, which
    can be useful inside shielded cleanup code).
    """
    async with _disconnect_event(request.scope, request.receive) as event:
        yield event


if fastapi is not None:
    # Ready-made alias: `async def handler(disconnected: Disconnected) -> ...`
    Disconnected = Annotated[asyncio.Event, fastapi.Depends(disconnected_event)]


class CancelOnDisconnectMiddleware:
    """Pure ASGI middleware: cancel the downstream app on client disconnect.

    App-wide and handler-agnostic — no signature requirements. Proxies the
    receive channel through a queue so both the watcher and the downstream
    app see every message, and publishes the shared per-request disconnect
    event that the decorator and `disconnected_event` dependency reuse.

    Caveat: the watcher reads eagerly, so request bodies are buffered in
    memory without backpressure. Fine for JSON APIs, wrong for large uploads
    — exempt those routes via `exclude_paths`.

    `exclude_paths` are regexes matched against the request path with
    `re.fullmatch`. `on_disconnect` (sync or async, called with the ASGI
    scope) runs after a premature disconnect cancelled a request — a hook
    for logging or metrics; it is not called on normal connection close.
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Sequence[str] = (),
        on_disconnect: OnDisconnect | None = None,
    ) -> None:
        self.app = app
        self.exclude_paths = [re.compile(pattern) for pattern in exclude_paths]
        self.on_disconnect = on_disconnect

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # An upstream instance already proxies this request; adding a second
        # queue layer would only cost, so nesting is a no-op.
        if (
            scope["type"] != "http"
            or _SCOPE_KEY in scope
            or any(pattern.fullmatch(scope["path"]) for pattern in self.exclude_paths)
        ):
            await self.app(scope, receive, send)
            return

        queue: asyncio.Queue[Message] = asyncio.Queue()
        response_complete = False

        async def tee() -> Message:
            message = await receive()
            queue.put_nowait(message)
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_complete
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                response_complete = True

        async def premature_disconnect() -> None:
            await event.wait()
            if response_complete:
                # A disconnect after the response is a normal connection
                # close (uvicorn reports one as soon as the response
                # completes); background tasks still running downstream
                # must not be cancelled.
                await asyncio.sleep(math.inf)

        async with (
            _disconnect_event(scope, tee) as event,
            _scope_cancelled_when(premature_disconnect) as guard,
        ):
            await self.app(scope, queue.get, guarded_send)

        if guard.cancelled_caught and self.on_disconnect is not None:
            outcome = self.on_disconnect(scope)
            if inspect.isawaitable(outcome):
                await outcome
