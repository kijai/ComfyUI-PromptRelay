"""Headless pipeline runner — proves the brain end-to-end without ComfyUI or the UI.

    python -m relay_studio.run sample_inputs.json
    python -m relay_studio.run sample_inputs.json --no-llm   # skip the model, run pipeline deterministically

The JSON may contain the user-inputs fields plus an optional "request" string.
"""

import argparse
import json
import os
import sys


def _out(s):
    # Windows consoles are often cp1252 — encode defensively so logging never crashes.
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


def _print_event(ev):
    kind = ev.get("kind")
    if kind == "thinking":
        _out(f"\n  [think] {ev['text'][:400]}")
    elif kind == "status":
        _out(f"  ... {ev['text']}")
    elif kind == "tool_call":
        _out(f"\n  $ {ev['name']}({json.dumps(ev.get('args', {}))[:160]})")
    elif kind == "tool_result":
        r = ev.get("result", {})
        tag = "FAIL " + r["error"] if r.get("error") else "OK " + (r.get("file", "ok"))
        _out(f"    -> {tag}")
    elif kind == "task":
        _out(f"  [{ev['done']}/{ev['total']}] {ev['title']}: {ev['status']}")
    elif kind == "media":
        it = ev["item"]
        _out(f"  + media ({it['kind']}): {it['file']}  [{it['label']}]")
    elif kind == "assistant":
        _out(f"\n[Dolly] {ev['text']}\n")
    elif kind == "final":
        _out(f"\n=== FINAL: {ev['file']} ===")
    elif kind == "error":
        _out(f"  !! {ev['text']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the studio pipeline headless.")
    ap.add_argument("inputs", help="user-inputs JSON file")
    ap.add_argument("--request", default=None, help="override the request text")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the model; run the pipeline deterministically")
    args = ap.parse_args(argv)

    with open(args.inputs, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    request = args.request or data.pop("request", None) or \
        "Turn my cat into a talking influencer."

    from . import agent
    from .session import Project, register
    project = register(Project(inputs=data, request=request))
    project.on_event(_print_event)
    print(f"Project {project.id}  ->  {project.dir}\n")

    if args.no_llm:
        agent.ensure_complete(project, request)
        project.save()
    else:
        agent.run(project, request)

    final = next((m for m in project.media if m.get("meta", {}).get("final")), None)
    if final:
        print(f"\nDone. Final video: {project.media_path(final['file'])}")
        return 0
    print("\nNo final video was produced.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
