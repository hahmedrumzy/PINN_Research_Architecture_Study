"""
Export this Claude Code conversation to readable markdown.

    py -3.13 kaggle/export_chat.py              # my replies + your messages
    py -3.13 kaggle/export_chat.py --assistant  # my replies only
"""
import glob, json, os, sys

SRC = glob.glob(os.path.join(
    os.path.expanduser("~"), ".claude", "projects",
    "c--Users-haziq--vscode-Registration-PINNs", "*.jsonl"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.md")
only_assistant = "--assistant" in sys.argv


def text_of(msg):
    """Pull plain text out of a message, skipping tool calls and results."""
    c = msg.get("content")
    if isinstance(c, str):
        return c.strip()
    if not isinstance(c, list):
        return ""
    parts = [b.get("text", "") for b in c
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


SKIP = ("<system-reminder>", "[SYSTEM NOTIFICATION", "<task-notification>",
        "<ide_opened_file>", "Caveat:", "<command-name>", "<local-command")

rows, seen = [], set()
for path in SRC:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except Exception:
                continue
            role = (e.get("message") or {}).get("role") or e.get("type")
            if role not in ("user", "assistant"):
                continue
            t = text_of(e.get("message") or {})
            if not t or any(t.lstrip().startswith(s) for s in SKIP):
                continue
            if only_assistant and role != "assistant":
                continue
            key = (role, t[:200])
            if key in seen:
                continue
            seen.add(key)
            rows.append((role, t))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("# Registration PINNs - conversation transcript\n\n")
    for role, t in rows:
        fh.write(f"\n---\n\n## {'You' if role == 'user' else 'Claude'}\n\n{t}\n")

words = sum(len(t.split()) for _, t in rows)
print(f"wrote {OUT}")
print(f"  {len(rows)} messages, ~{words:,} words, {os.path.getsize(OUT)/1024:.0f} KB")
