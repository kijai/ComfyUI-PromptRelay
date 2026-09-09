"""Standalone dev server — runs the studio API + UI WITHOUT ComfyUI.

Mounts the same routes used inside ComfyUI by injecting a fake `server.PromptServer`,
so the full UI (create project -> chat -> media -> final video) can be exercised
offline. Image generation still needs real ComfyUI; the cat pipeline does not.

    python -m relay_studio._standalone           # http://localhost:8189/promptrelay/studio/ui
"""

import os
import sys
import types

from aiohttp import web

# Allow running as a plain script (python .../_standalone.py): ensure the package's
# parent dir is importable so `import relay_studio` resolves.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


def build_app():
    routes = web.RouteTableDef()
    fake = types.ModuleType("server")
    fake.PromptServer = type("PromptServer", (), {})
    fake.PromptServer.instance = types.SimpleNamespace(routes=routes)
    sys.modules["server"] = fake

    # import the package now that `server` is fakeable
    import relay_studio
    relay_studio.register_routes()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes(routes)
    return app


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8189
    print(f"Studio dev server: http://localhost:{port}/promptrelay/studio/ui")
    web.run_app(build_app(), host="127.0.0.1", port=port)
