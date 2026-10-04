#!/usr/bin/env python3
"""Point Claude Code at this kit, or back at DeepSeek - one flip either way.

Claude Code takes its endpoint, its key and its model from the `env` block of
its own settings file: `$CLAUDE_CONFIG_DIR/settings.json`, or
`~/.claude/settings.json` on the usual install. That is how this machine is
currently pointed at DeepSeek's Anthropic-compatible endpoint, and this kit
speaks the same Messages API on http://127.0.0.1:8888 - so the two are one
settings edit apart, and this is that edit.

    python3 tools/claude_toggle.py

No arguments: whichever profile is live, the other is written. The kit's `.env`
is found from this file's location, so it can be run from anywhere. Nothing
here needs the venv and nothing is imported from the engine; it is a file edit,
so plain python3 runs it on either system.

The two profiles
    deepseek   DeepSeek V4 Flash over https://api.deepseek.com/anthropic.
               Captured from the live settings the first time this runs while
               that profile is active, because the token in it is a credential
               that belongs in neither this file nor the repo. The capture is
               kept beside settings.json as `simplex-claude-toggle.json`, mode
               600, and never printed.
    local      this kit. Built from `.env` on every run - MODEL_ID, HOST, PORT
               and CONTEXT_SIZE - so switching quant, port or context does not
               need this script to know what the model of the day is.

Both profiles write the same ten keys (OWNED_KEYS below) and nothing else.
Every other setting in the file - the theme, hooks, permissions, and any `env`
key this script does not know - is read and written back untouched. A key a
profile does not carry is removed rather than left behind, so the profile in
the file is exactly the profile and not the profile plus what the other one
left: DeepSeek's `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (786432 - a 1M-token window)
disappears when this kit, 229376 tokens here, takes over, and the kit's
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` disappears when DeepSeek comes back.

If the file is on neither profile - a third provider was typed into it, or
there is no endpoint at all - this says which URL it found and stops rather
than guessing a direction. `--local` or `--deepseek` is how you say which way
to go from there.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# The `.env` rules - KEY=value, a trailing comment, quotes stripped, CRLF
# tolerated - and the colour policy already exist once in the kit, in
# tools/cli.py. They are borrowed rather than written a fourth time: a second
# reader is how the Windows and Linux launchers came to disagree about what
# `.env` says. cli imports the standard library and dsh.py, which does the
# same, so this stays a script that runs wherever python3 does.
import cli                                                     # noqa: E402

# The keys the two profiles own. Everything else in `env`, and every other
# setting in the document, is somebody else's and is written back untouched.
OWNED_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_EFFORT_LEVEL",
)

# The captured DeepSeek profile, written beside the settings file it was read
# from. It holds a live API token: mode 600, and `shown()` is the only way it
# is ever rendered, by its ends.
SNAPSHOT_NAME = "simplex-claude-toggle.json"

# Whose endpoint is this? The base URL is the one key the two profiles cannot
# agree on, so it is what answers the question.
DEEPSEEK_DOMAINS = ("deepseek.com",)
LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

DEFAULT_PORT = 8888
LOCAL_TOKEN = "local"       # the kit has no key; clients still have to send one

# Claude Code reads these out of its own environment as well as out of the
# file, and `settings.json` is not the only place on a machine they can come
# from. Worth saying out loud when one of them is already exported.
SHADOWING = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL")

COLUMN = 32                 # widest owned key is 30 characters


class Fail(Exception):
    """Something the person running this can fix, said with what to do."""


# ---------------------------------------------------------------- output ----
def out(msg: str = "") -> None:
    print(msg, flush=True)


def line(label: str, text: str, colour=None) -> None:
    out((colour or cli.dim)(f"  {label:<12}") + text)


def warn(text: str) -> None:
    line("warn", text, cli.yellow)


def shown(key: str, value) -> str:
    """A token by its ends and nothing else.

    This output goes to a terminal, into a scrollback and quite possibly into a
    paste; the DeepSeek key is a live credential and has no business in any of
    them.
    """
    if value is None:
        return "(absent)"
    text = str(value)
    if ("TOKEN" in key or "API_KEY" in key) and len(text) > 12:
        return f"{text[:6]}...{text[-4:]}"
    return text


def describe(mode: str | None, env: dict) -> str:
    """One line: which profile, its base URL, the model it names."""
    if mode is None:
        return ("neither profile  "
                + shown("ANTHROPIC_BASE_URL", env.get("ANTHROPIC_BASE_URL")))
    base = env.get("ANTHROPIC_BASE_URL") or "(no base url)"
    model = env.get("ANTHROPIC_MODEL") or "(no model)"
    return f"{mode:<9}{base}  model {model}"


# -------------------------------------------------------------- profiles ----
def config_dir() -> Path:
    """Where Claude Code keeps settings.json.

    It reads CLAUDE_CONFIG_DIR, which is the whole reason to ask rather than
    assume; `Path.home()` is %USERPROFILE% on Windows, so the fallback is right
    on both systems.
    """
    override = (os.environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    return Path(override).expanduser() if override else Path.home() / ".claude"


def endpoint(cfg: dict) -> str:
    """This kit's Anthropic endpoint as a client has to write it.

    No `/v1` on the end: the Anthropic SDKs append `/v1/messages` themselves,
    and Claude Code does the same. A wildcard bind is not an address to point a
    client at - `0.0.0.0` is not even reachable on Windows - so it becomes
    loopback, which is where the server is listening either way.
    """
    host = (cfg.get("HOST") or "").strip() or "127.0.0.1"
    if host in ("0.0.0.0", "*", "::", "[::]"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"                     # an IPv6 literal needs brackets
    return f"http://{host}:{cli.env_int(cfg, 'PORT', DEFAULT_PORT)}"


def local_hosts(cfg: dict) -> set[str]:
    """Everything that means "the kit" when it turns up in a base URL."""
    hosts = {h.lower() for h in LOOPBACK}
    host = urlparse(endpoint(cfg)).hostname
    if host:
        hosts.add(host.lower())
    return hosts


def mode_of(env: dict, hosts: set[str]) -> str | None:
    """Which profile a settings file is on, or None for neither.

    Judged by the base URL alone - it is the one key the two profiles cannot
    share, and it is the one a person would have typed in by hand if they set
    this up themselves. None is a state with no safe guess in it, so the caller
    stops instead of writing.
    """
    base = str(env.get("ANTHROPIC_BASE_URL") or "").strip()
    if not base:
        return None
    # A value typed without a scheme - "127.0.0.1:8888" - is a base URL too, and
    # urlparse reads that first label as a scheme and hands back no host at all.
    parsed = urlparse(base if "//" in base else "//" + base)
    host = (parsed.hostname or "").lower()
    if any(host == d or host.endswith("." + d) for d in DEEPSEEK_DOMAINS):
        return "deepseek"
    return "local" if host in hosts else None


def kit_profile(cfg: dict, notes: list[str]) -> tuple[dict, str]:
    """The kit's profile, built from `.env` on every run. Returns (env, host).

    The base URL, the dummy key and the model id are the README's Claude Code
    recipe. The three model aliases are pointed at the same id rather than left
    alone, so a session that asks for the small or the big model stays on this
    machine instead of naming a DeepSeek model: the server serves whatever it
    loaded whatever name it is sent, so a stale name would work and be a lie in
    the model picker.
    """
    model = (cfg.get("MODEL_ID") or "").strip().lower()
    if not model:
        directory = (cfg.get("MODEL_DIR") or "").strip().strip('"')
        model = Path(directory).name.lower() if directory else ""
    if not model:
        raise Fail("MODEL_ID is not set in .env and MODEL_DIR names no folder, so "
                   "there is no model id to give Claude Code - run `simplex setup`")

    base = endpoint(cfg)
    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": LOCAL_TOKEN,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
    }
    context = cli.env_int(cfg, "CONTEXT_SIZE", 0)
    if context:
        # Claude Code assumes a 200k window unless told otherwise, and this kit
        # is not always 200k - here it is larger, on a small card smaller.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context)
    else:
        notes.append("CONTEXT_SIZE is not in .env, so Claude Code keeps its own "
                     "window assumption and may compact at the wrong point")
    return env, urlparse(base).hostname or "127.0.0.1"


# -------------------------------------------------------------- settings ----
def read_settings(path: Path) -> tuple[dict, dict]:
    """(document, env). The document is kept whole and handed back on write.

    A flip is an edit to one key of one object. The theme, the hooks, the
    permission rules, the env keys this script has never heard of: all of it
    goes back into the file exactly as it came out.
    """
    if not path.is_file():
        raise Fail(f"{path} does not exist - start Claude Code once, or point "
                   f"--settings at the file you mean")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise Fail(f"{path} cannot be read ({e})")
    except ValueError as e:
        raise Fail(f"{path} is not valid JSON ({e}) - Claude Code will not read "
                   f"it either, so fix or remove it and run this again")
    if not isinstance(doc, dict):
        raise Fail(f"{path} is not a JSON object")
    env = doc.get("env", {})
    if not isinstance(env, dict):
        raise Fail(f'"env" in {path} is not an object, so there is nowhere to '
                   f'write a profile - fix it by hand first')
    return doc, env


def write_settings(path: Path, doc: dict) -> None:
    """2-space JSON, the layout Claude Code writes.

    Written next to the original and renamed over it, so an interrupted run
    cannot leave a settings.json that will not parse. A symlinked settings.json
    - people do keep this file in a dotfiles repo - is written through rather
    than replaced: os.replace would turn the link into a second file that the
    repo never hears about.
    """
    real = path.resolve()
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    tmp = real.with_name(real.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, real)


# -------------------------------------------------------------- snapshot ----
def read_snapshot(path: Path) -> tuple[dict, str]:
    """(the DeepSeek env as last captured, why it could not be read).

    Only keys this script owns come back, so a hand-edited file cannot smuggle
    anything into settings.json on the next flip. An unreadable one is not
    fatal in either direction: while DeepSeek is live it is about to be
    overwritten anyway, and while it is not, the caller's own "not captured
    yet" is the honest answer - so the reason travels back rather than being
    raised here.
    """
    if not path.is_file():
        return {}, ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, (f"{path} will not parse as JSON ({e}) - delete it, or flip to "
                    f"DeepSeek once with DeepSeek still live, and it is captured "
                    f"again")
    env = doc.get("env") if isinstance(doc, dict) else None
    if not isinstance(env, dict):
        return {}, ""
    return {k: v for k, v in env.items() if k in OWNED_KEYS}, ""


def write_snapshot(path: Path, env: dict, base_url: str) -> None:
    """Capture the live DeepSeek env, mode 600 from the moment it exists.

    The file is created with that mode rather than chmod-ed afterwards, so
    there is no instant at which the token is on disk and world-readable.
    """
    doc = {
        "tool": "tools/claude_toggle.py",
        "note": ("The Claude Code profile this script puts back when it flips "
                 "away from DeepSeek. It holds an API token, which is why it is "
                 "beside settings.json and not in the kit's repo."),
        "captured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "env": {k: env[k] for k in OWNED_KEYS if k in env},
    }
    text = json.dumps(doc, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- checks ----
def port_open(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=0.4).close()
        return True
    except OSError:
        return False


def project_override() -> tuple[Path, str] | None:
    """A project settings file that also sets the endpoint, if this folder has one.

    Claude Code reads project settings after the user ones, so such a file wins
    and a flip would look like it did nothing at all. Only worth a warning -
    it is somebody's deliberate choice, not a mistake to correct.
    """
    folder = Path.cwd()
    for name in (".claude/settings.local.json", ".claude/settings.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        env = doc.get("env") if isinstance(doc, dict) else None
        if isinstance(env, dict) and any(k in env for k in OWNED_KEYS):
            return path, str(env.get("ANTHROPIC_BASE_URL") or "")
    return None


# ------------------------------------------------------------------ flip ----
def flip(to: str | None, dry_run: bool, settings: Path, snapshot: Path,
         cfg: dict) -> int:
    doc, env = read_settings(settings)
    here = mode_of(env, local_hosts(cfg))
    target = to or {"deepseek": "local", "local": "deepseek"}.get(here)
    if target is None:
        raise Fail(f"{settings} is on neither profile - ANTHROPIC_BASE_URL is "
                   f"{shown('ANTHROPIC_BASE_URL', env.get('ANTHROPIC_BASE_URL'))}. "
                   f"Say which way to go: --local or --deepseek")

    # What the DeepSeek side says, now and as last captured. A live one wins:
    # it is the file's own words, and re-capturing is what picks up a rotated
    # token or a changed model without anyone editing this script.
    saved, why = read_snapshot(snapshot)
    live = ({k: v for k, v in env.items() if k in OWNED_KEYS}
            if here == "deepseek" else None)
    capture = live is not None and live != saved

    notes: list[str] = []
    host, port = "", 0
    if target == "local":
        want, host = kit_profile(cfg, notes)
        port = cli.env_int(cfg, "PORT", DEFAULT_PORT)
    else:
        want = live or saved
        if not want:
            raise Fail(why or (
                "the DeepSeek profile has not been captured yet, so there is "
                "nothing to put back - run this once while Claude Code is on "
                f"DeepSeek and it is saved to {snapshot}"))

    new_env = dict(env)
    for key in OWNED_KEYS:
        if key in want:
            new_env[key] = want[key]
        else:
            new_env.pop(key, None)              # the profile is the whole truth
    changes = [(k, env.get(k), new_env.get(k)) for k in OWNED_KEYS
               if env.get(k) != new_env.get(k)]

    # ---- the report, before anything is touched ---------------------------
    out()
    line("settings", str(settings))
    line("on", describe(here, env))
    if here != target:
        line("->", describe(target, want))
    out()
    if changes:
        for key, was, now in changes:
            out(f"    {key:<{COLUMN}} {shown(key, was)} -> {shown(key, now)}")
    else:
        word = (f"already the {target} profile" if here == target
                else f"the {target} profile differs in no key this script owns")
        out(cli.green("  settings.json already says that")
            + cli.dim(f" - {word}"))
    out()

    # ---- write ------------------------------------------------------------
    if not dry_run:
        if capture:
            write_snapshot(snapshot, live, str(env.get("ANTHROPIC_BASE_URL") or ""))
        if changes:
            doc["env"] = new_env
            write_settings(settings, doc)

    if capture:
        saved_what = f"the deepseek env in {snapshot}"
        line("would save" if dry_run else "saved", saved_what, cli.green)
    if dry_run:
        line("dry run", "nothing written", cli.yellow)
    elif changes:
        line("done", f"{len(changes)} key(s) changed", cli.green)

    # ---- what to know next ------------------------------------------------
    if target == "local" and not port_open(host, port):
        warn(f"nothing is answering on {host}:{port} - start the kit "
             f"(./linux/simplex start) or claude will fail on every request")
    override = project_override()
    if override:
        warn(f"{override[0]} also sets the endpoint "
             f"({override[1] or 'with no base URL'}), and project settings are "
             f"read after user ones - a session started in {Path.cwd()} uses "
             f"that file, not this one")
    exported = [k for k in SHADOWING if os.environ.get(k)]
    if exported:
        warn(f"{', '.join(exported)} is set in this shell as well - check that a "
             f"claude started from here is meant to use it")
    if why and not capture:
        warn(why)
    for note in notes:
        line("note", note)
    if changes and not dry_run and target == "local":
        line("note", "a session that is already running has read the old endpoint; "
                     "start a new one")
    out()
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="claude_toggle.py",
        description="Point Claude Code at this kit's Anthropic endpoint, or back "
                    "at DeepSeek. With no arguments it flips to whichever "
                    "profile is not live.",
        epilog="Only the keys this script owns are touched; everything else in "
               "settings.json is written back as it was.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local", dest="to", action="store_const", const="local",
                   help="end on this kit's endpoint, whichever profile is live")
    g.add_argument("--deepseek", dest="to", action="store_const", const="deepseek",
                   help="end on the DeepSeek profile captured earlier")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would change and write nothing")
    p.add_argument("--settings", metavar="FILE",
                   help="the settings.json to edit, and where its capture is kept "
                        "(default: $CLAUDE_CONFIG_DIR/settings.json, else "
                        "~/.claude/settings.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    settings = (Path(a.settings).expanduser() if a.settings
                else config_dir() / "settings.json")
    try:
        return flip(a.to, a.dry_run, settings, settings.with_name(SNAPSHOT_NAME),
                    cli.read_env())
    except Fail as e:
        print(cli.red("  error  ") + str(e), file=sys.stderr, flush=True)
        return 1
    except OSError as e:
        # The report above has already been printed by then, so say what did
        # not land rather than letting a traceback be the last word.
        print(cli.red("  error  ") + f"nothing was written ({e})",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
