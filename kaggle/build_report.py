"""
Combine FINDINGS.md and the conversation transcript into one document.

    py -3.13 kaggle/build_report.py

Writes REPORT.md at the repo root: the findings first (that is what a reader
wants), then the full working transcript as an appendix (that is the evidence
trail, including the wrong turns).
"""
import os
import subprocess
import sys
import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = os.path.join(REPO, "FINDINGS.md")
CHAT = os.path.join(REPO, "kaggle", "conversation.md")
OUT = os.path.join(REPO, "REPORT.md")

# refresh the transcript first so the appendix is current
subprocess.run([sys.executable, os.path.join(REPO, "kaggle", "export_chat.py")],
               capture_output=True, text=True)

findings = open(FINDINGS, encoding="utf-8").read()
chat = open(CHAT, encoding="utf-8").read()

# the transcript file has its own H1; demote it and everything under it so the
# combined document has a single heading hierarchy
lines = []
for line in chat.split("\n"):
    if line.startswith("# Registration PINNs"):
        continue
    if line.startswith("#"):
        line = "#" + line          # H2 -> H3 etc, keeps the appendix subordinate
    lines.append(line)
chat_body = "\n".join(lines)

stamp = datetime.date.today().isoformat()
n_msgs = chat.count("\n## ")
n_words = len(chat.split())

header = """# Physics-informed point-set registration — report

Generated """ + stamp + """ from `FINDINGS.md` and the working transcript.

**Part I — Findings.** Measured results, with the raw data in `kaggle/results/`
and every generator and experiment script in this repo.

**Part II — Working transcript.** The full session, """ + f"{n_msgs}" + """ messages,
""" + f"{n_words:,}" + """ words. Included unedited, so the wrong turns are visible
alongside the results: the abandoned width-versus-complexity sweep, the strain
fix that measured correctly and broke training, the Stage-2 runs discarded after
being traced to that fix, and the GPU quota spent on all three.

---

"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(header)
    f.write("# Part I — Findings\n\n")
    # drop the transcript's own H1 line from findings, keep the rest
    fbody = findings.split("\n", 1)[1] if findings.startswith("# ") else findings
    f.write(fbody.lstrip("\n"))
    f.write("\n\n---\n\n# Part II — Working transcript\n\n")
    f.write(chat_body.lstrip("\n"))

size = os.path.getsize(OUT)
print("wrote " + OUT)
print("  " + str(size // 1024) + " KB")
print("  Part I  : findings (" + str(len(findings.split())) + " words)")
print("  Part II : transcript (" + str(n_words) + " words, " + str(n_msgs) + " messages)")
