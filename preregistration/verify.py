"""Checks the pre-registration hashes and compares the archived v2 notebooks with the ones in notebooks/.

locked_files/ holds the files exactly as they were when their hashes were recorded. The notebooks in notebooks/
contain the same code; only comments, docstrings, markdown and the characters of printed messages were edited later.
The second check parses both versions and compares the program structure with those parts removed.

Run from the repository root:  python preregistration/verify.py
"""
import ast, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, "locked_files")
NB = os.path.join(os.path.dirname(HERE), "notebooks")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_hashes(listing, rename=None):
    ok = True
    for line in open(os.path.join(LOCK, listing), encoding="utf-8"):
        digest, name = line.split(maxsplit=1); name = name.strip().lstrip("*")
        path = os.path.join(LOCK, (rename or {}).get(name, name))
        good = os.path.exists(path) and sha(path) == digest
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {name}")
    return ok


SYMS = [(" \u2014 ", " - "), ("\u2014", "-"), (" \u2192 ", " -> "), ("\u2192", "->"), ("\u2550", "="), ("\u2500", "-"),
        ("\u2026", "...")]


def structure(src):
    """AST of a code cell without docstrings; string constants are normalised the same way for both versions."""
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and n.body:
            e = n.body[0]
            if isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str):
                e.value.value = ""
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            for a, b in SYMS: v = v.replace(a, b)
            n.value = re.sub(r"^[AB]\d{1,2}: ", "", v)
    return ast.dump(tree)


def code_cells(path):
    cells = json.load(open(path, encoding="utf-8"))["cells"]
    return [structure("".join(c["source"])) for c in cells if c["cell_type"] == "code"]


print("v2 pre-registration (original text, locked 2026-10-08):")
ok1 = check_hashes("preregistration_v2.sha256", {"preregistration_v2.md": "preregistration_v2_locked_2026-10-08.md"})
print("Amendment 1 (locked before Stage 2):")
ok2 = check_hashes("preregistration_v2_amendment1.sha256")
print("Archived notebooks vs notebooks/ (same code):")
ok3 = True
for locked, current in (("tahap1_pengembangan/egvw_part5_v2_A.ipynb", "v2_stage1/v2_stage1_A.ipynb"),
                        ("tahap1_pengembangan/egvw_part5_v2_B.ipynb", "v2_stage1/v2_stage1_B.ipynb"),
                        ("tahap2_konfirmasi/t2-kan-a.ipynb", "v2_stage2/stage2_kan_A.ipynb"),
                        ("tahap2_konfirmasi/t2-kan-b.ipynb", "v2_stage2/stage2_kan_B.ipynb"),
                        ("tahap2_konfirmasi/t2-comp.ipynb", "v2_stage2/stage2_comparators.ipynb")):
    same = code_cells(os.path.join(LOCK, locked)) == code_cells(os.path.join(NB, current))
    ok3 &= same
    print(f"  {'OK  ' if same else 'DIFF'} {current}")
print("all checks passed" if ok1 and ok2 and ok3 else "SOME CHECKS FAILED")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
