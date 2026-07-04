import asyncio

from fastapi import FastAPI

from fastapi_disconnect import (
    CancelOnDisconnectMiddleware,
    Disconnected,
    cancel_on_disconnect,
)

app = FastAPI()


async def long_flow(tag: str) -> str:
    try:
        for i in range(10):
            await asyncio.sleep(1)
            print(f"[{tag}] working... {i}")
        return "OK"
    except asyncio.CancelledError:
        print(f"[{tag}] cancelled")
        raise


@app.get("/decorator")
@cancel_on_disconnect
async def decorator_handler() -> str:
    return await long_flow("decorator")


@app.get("/cooperative")
async def cooperative_handler(disconnected: Disconnected) -> str:
    for i in range(10):
        if disconnected.is_set():
            print(f"[cooperative] client left, stopping after step {i}")
            return f"partial after {i} steps"
        await asyncio.sleep(1)
        print(f"[cooperative] working... {i}")
    return "OK"


# The middleware cancels everything under it; mount a sub-app to scope it.
protected = FastAPI()
protected.add_middleware(CancelOnDisconnectMiddleware)
app.mount("/middleware", protected)


@protected.get("/")
async def middleware_handler() -> str:
    return await long_flow("middleware")
