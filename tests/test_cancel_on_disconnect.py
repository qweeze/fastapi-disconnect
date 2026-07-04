import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from fastapi_disconnect import (
    CancelOnDisconnectMiddleware,
    Disconnected,
    cancel_on_disconnect,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def http_scope(method: str = "GET", path: str = "/", body: bytes = b"") -> Scope:
    headers = [(b"host", b"test")]
    if body:
        headers += [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": headers,
        "client": ("test", 0),
        "server": ("test", 80),
    }


def scripted_receive(script: list[tuple[float, Message]]) -> Receive:
    """Yield each message after its delay, then block forever."""
    pending = list(script)

    async def receive() -> Message:
        if pending:
            delay, message = pending.pop(0)
            await asyncio.sleep(delay)
            return message
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return receive


def collector() -> tuple[list[Message], Send]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    return sent, send


def body_message(body: bytes = b"") -> Message:
    return {"type": "http.request", "body": body, "more_body": False}


DISCONNECT: Message = {"type": "http.disconnect"}


async def test_middleware_cancels_on_disconnect() -> None:
    cancelled = asyncio.Event()
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware)

    @app.get("/")
    async def handler() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert cancelled.is_set()
    assert sent == []  # nothing was sent to the dead client


async def test_middleware_passes_body_and_response_through() -> None:
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware)

    @app.post("/echo")
    async def echo(data: dict[str, Any]) -> dict[str, Any]:
        return data

    payload = b'{"a": 1}'
    sent, send = collector()
    receive = scripted_receive([(0, body_message(payload))])
    async with asyncio.timeout(2):
        await app(http_scope("POST", "/echo", body=payload), receive, send)

    assert sent[0]["status"] == 200
    assert json.loads(sent[1]["body"]) == {"a": 1}


async def test_decorator_cancels_and_responds_499() -> None:
    cancelled = asyncio.Event()
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert cancelled.is_set()
    assert sent[0]["status"] == 499


async def test_decorator_normal_completion() -> None:
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler() -> str:
        await asyncio.sleep(0.01)
        return "hi"

    sent, send = collector()
    receive = scripted_receive([(0, body_message())])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b'"hi"'


async def test_decorator_exception_passthrough() -> None:
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler() -> str:
        raise HTTPException(status_code=418, detail="teapot")

    sent, send = collector()
    receive = scripted_receive([(0, body_message())])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert sent[0]["status"] == 418


async def test_async_cleanup_on_cancellation_requires_shield() -> None:
    """Cancellation is level-triggered: unshielded awaits in cleanup are
    re-cancelled at the first checkpoint; a shielded scope runs to completion."""
    events: list[str] = []
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            try:
                await asyncio.sleep(1)
                events.append("unshielded done")
            except asyncio.CancelledError:
                events.append("unshielded cancelled")
            with anyio.CancelScope(shield=True):
                await asyncio.sleep(0.01)
                events.append("shielded done")
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert events == ["unshielded cancelled", "shielded done"]
    assert sent[0]["status"] == 499


async def test_dependency_cooperative_exit() -> None:
    """The event enables graceful early exit: no cancellation, partial result."""
    app = FastAPI()

    @app.get("/")
    async def handler(disconnected: Disconnected) -> str:
        for _ in range(100):
            if disconnected.is_set():
                return "partial"
            await asyncio.sleep(0.02)
        return "full"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b'"partial"'


async def test_dependency_composes_with_decorator() -> None:
    """Dependency and decorator share one watcher; both signals fire."""
    observed: list[bool] = []
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler(disconnected: Disconnected) -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            observed.append(disconnected.is_set())
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert observed == [True]  # cancelled AND the event was set
    assert sent[0]["status"] == 499


async def test_full_stack_composes() -> None:
    """Middleware + decorator + dependency on one route: the single shared
    receive reader means every mechanism still fires — no lost disconnect."""
    observed: list[bool] = []
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware)

    @app.get("/")
    @cancel_on_disconnect
    async def handler(disconnected: Disconnected) -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            observed.append(disconnected.is_set())
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert observed == [True]
    # Depending on which cancellation lands first, either nothing was sent
    # or the decorator's 499 was; a 200 would mean a mechanism misfired.
    statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert statuses in ([], [499])


async def test_nested_middleware_is_noop() -> None:
    """A second middleware instance (e.g. outer app + mounted sub-app) must
    pass through instead of stacking a queue layer that never gets fed."""
    cancelled = asyncio.Event()
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware)
    app.add_middleware(CancelOnDisconnectMiddleware)

    @app.post("/echo")
    async def echo(data: dict[str, Any]) -> dict[str, Any]:
        return data

    @app.get("/slow")
    async def slow() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "OK"

    payload = b'{"a": 1}'
    sent, send = collector()
    receive = scripted_receive([(0, body_message(payload))])
    async with asyncio.timeout(2):  # would time out if the body never arrived
        await app(http_scope("POST", "/echo", body=payload), receive, send)
    assert sent[0]["status"] == 200

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(path="/slow"), receive, send)
    assert cancelled.is_set()


def test_decorator_rejects_sync_handler() -> None:
    with pytest.raises(TypeError, match="async def"):

        @cancel_on_disconnect  # type: ignore[arg-type]  # deliberate: sync handler
        def handler() -> str:
            return "OK"


async def test_decorator_warns_on_streaming_response() -> None:
    """The guard exits when the handler returns; a StreamingResponse body
    runs after that, uncancelled — the decorator should say so."""
    app = FastAPI()

    @app.get("/")
    @cancel_on_disconnect
    async def handler() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            yield b"chunk"

        return StreamingResponse(gen())

    sent, send = collector()
    receive = scripted_receive([(0, body_message())])
    with pytest.warns(RuntimeWarning, match="StreamingResponse"):
        async with asyncio.timeout(2):
            await app(http_scope(), receive, send)

    assert sent[0]["status"] == 200


async def test_middleware_exclude_paths() -> None:
    done = asyncio.Event()
    cancelled = asyncio.Event()
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware, exclude_paths=[r"/excluded", r"/files/.*"])

    @app.get("/excluded")
    async def excluded() -> str:
        await asyncio.sleep(0.2)
        done.set()
        return "OK"

    @app.get("/covered")
    async def covered() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "OK"

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(path="/excluded"), receive, send)
    assert done.is_set()  # ran to completion despite the disconnect
    assert sent[0]["status"] == 200

    sent, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(path="/covered"), receive, send)
    assert cancelled.is_set()


async def test_middleware_on_disconnect_callback() -> None:
    seen: list[str] = []

    async def record(scope: Scope) -> None:
        seen.append(scope["path"])

    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware, on_disconnect=record)

    @app.get("/slow")
    async def slow() -> str:
        await asyncio.sleep(10)
        return "OK"

    @app.get("/fast")
    async def fast() -> str:
        return "OK"

    _, send = collector()
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(path="/slow"), receive, send)
    assert seen == ["/slow"]

    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(path="/fast"), receive, send)
    assert seen == ["/slow"]  # a close after the response is not a disconnect

    # Sync callbacks work too.
    seen_sync: list[str] = []
    app2 = FastAPI()
    app2.add_middleware(
        CancelOnDisconnectMiddleware,
        on_disconnect=lambda scope: seen_sync.append(scope["path"]),
    )

    @app2.get("/slow")
    async def slow2() -> str:
        await asyncio.sleep(10)
        return "OK"

    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app2(http_scope(path="/slow"), receive, send)
    assert seen_sync == ["/slow"]


async def test_decorator_direct_call_runs_unguarded() -> None:
    @cancel_on_disconnect
    async def plain(x: int) -> int:
        return x + 1

    assert await plain(1) == 2


async def test_middleware_spares_background_tasks() -> None:
    """A disconnect arriving after the response is complete is a normal
    connection close, not a premature disconnect: background tasks (which
    run downstream, after the response is sent) must not be cancelled.
    Servers make this common: uvicorn reports http.disconnect on receive()
    as soon as the response completes."""
    from fastapi import BackgroundTasks

    done = asyncio.Event()
    app = FastAPI()
    app.add_middleware(CancelOnDisconnectMiddleware)

    async def slow_side_effect() -> None:
        await asyncio.sleep(0.2)
        done.set()

    @app.get("/")
    async def handler(background: BackgroundTasks) -> str:
        background.add_task(slow_side_effect)
        return "OK"

    sent, send = collector()
    # Disconnect arrives right after the response, while the task still runs.
    receive = scripted_receive([(0, body_message()), (0.05, DISCONNECT)])
    async with asyncio.timeout(2):
        await app(http_scope(), receive, send)

    assert sent[0]["status"] == 200
    assert done.is_set()
