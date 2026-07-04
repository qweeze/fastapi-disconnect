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
        # Join so cleanup is complete and real watcher failures surface.
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


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
            with contextlib.suppress(asyncio.CancelledError):
                await watcher


def cancel_on_disconnect[**P, R](
    handler: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R | Response]]:
    """Cancel the request handler as soon as the client disconnects.

    The handler does not need to declare a `Request` parameter. The wrapper
    advertises one extra keyword-only `Request` parameter via
    `__signature__`; FastAPI injects it and the wrapper strips it off before
    calling the handler. If the handler declares its own `Request` parameter,
    that one is reused instead (FastAPI injects a request into only one
    parameter per endpoint). `Request`-annotated params never appear in
    OpenAPI. Called directly (e.g. in unit tests), the handler runs
    unguarded. Apply below the route decorator: `@app.get(...)` first,
    this second.

    Scope of the guard: it starts when the handler is invoked — dependencies
    and request parsing have already run by then — and while it is active the
    disconnect watcher owns the receive channel, so the handler must not read
    the raw body (`request.body()` / `request.stream()`; FastAPI-parsed body
    params are fine). Use `CancelOnDisconnectMiddleware` for either case.
    """
    if not inspect.iscoroutinefunction(handler):
        raise TypeError(
            f"@cancel_on_disconnect requires an async def handler, got {handler!r}: "
            "sync handlers run in a threadpool and cannot be cancelled"
        )

    # eval_str resolves stringified annotations with the handler's globals,
    # which FastAPI could not do itself through a wrapper from another module.
    signature = inspect.signature(handler, eval_str=True)
    if _REQUEST_PARAM in signature.parameters:
        raise TypeError(f"{handler.__qualname__}: parameter name {_REQUEST_PARAM!r} is reserved")
    own_request_param = next(
        (
            name
            for name, param in signature.parameters.items()
            if isinstance(param.annotation, type) and issubclass(param.annotation, Request)
        ),
        None,
    )

    @functools.wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | Response:
        if own_request_param is not None:  # stays in kwargs: the handler wants it
            request = kwargs.get(own_request_param) or next(
                (arg for arg in args if isinstance(arg, Request)), None
            )
        else:
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

    if own_request_param is None:
        # Advertise the synthetic parameter, keeping it ahead of any **kwargs
        # (keyword-only params must precede VAR_KEYWORD in a signature).
        parameters = list(signature.parameters.values())
        position = next(
            (
                index
                for index, param in enumerate(parameters)
                if param.kind is inspect.Parameter.VAR_KEYWORD
            ),
            len(parameters),
        )
        parameters.insert(
            position,
            inspect.Parameter(
                _REQUEST_PARAM, kind=inspect.Parameter.KEYWORD_ONLY, annotation=Request
            ),
        )
        signature = signature.replace(parameters=parameters)
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
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


def __getattr__(name: str) -> Any:  # pragma: no cover - reached only without fastapi
    if name == "Disconnected":
        raise ImportError(
            "fastapi_disconnect.Disconnected requires the 'fastapi' package; "
            "the rest of the library works with plain starlette"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CancelOnDisconnectMiddleware:
    """Pure ASGI middleware: cancel the downstream app on client disconnect.

    App-wide and handler-agnostic — no signature requirements. Proxies the
    receive channel through a queue so both the watcher and the downstream
    app see every message, and publishes the shared per-request disconnect
    event that the decorator and `disconnected_event` dependency reuse.

    Caveat: with the default unbounded queue the watcher reads eagerly, so
    request bodies are buffered in memory without backpressure. Set
    `queue_size` to bound the buffer (it counts server-sized body chunks,
    not bytes): backpressure is restored, at the cost of disconnect
    detection pausing while the queue is full. Alternatively exempt upload
    routes via `exclude_paths`, or enforce a request-size limit upstream.

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
        queue_size: int = 0,
    ) -> None:
        self.app = app
        self.exclude_paths = [re.compile(pattern) for pattern in exclude_paths]
        self.on_disconnect = on_disconnect
        self.queue_size = queue_size

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

        queue: asyncio.Queue[Message] = asyncio.Queue(self.queue_size)
        response_complete = False

        async def tee() -> Message:
            message = await receive()
            await queue.put(message)
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
