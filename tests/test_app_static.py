"""Static integrity checks on app.py — the tests that would have caught
every deployment failure so far: missing modules, missing dependencies,
undefined functions, missing tooltip keys, duplicate widget keys."""
import ast
import builtins
import importlib
import pathlib
import py_compile
import re

import pytest

import conftest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
MODULES = ["app", "model", "data", "journal", "auth"]


def _tree(path):
    return ast.parse(path.read_text())


def test_all_files_compile():
    for m in MODULES:
        py_compile.compile(str(ROOT / f"{m}.py"), doraise=True)


def test_every_called_function_is_defined_or_imported():
    tree = _tree(APP)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, ast.FunctionDef):
                for a in node.args.args + node.args.kwonlyargs:
                    defined.add(a.arg)
        elif isinstance(node, ast.Import):
            defined.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            defined.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.withitem) and node.optional_vars:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    missing = sorted(called - defined)
    assert not missing, f"called but never defined/imported: {missing}"


def test_cross_module_imports_resolve():
    """Catches the 'data.py contained journal.py' class of failure."""
    exports = {}
    for m in MODULES:
        names = set()
        for node in ast.walk(_tree(ROOT / f"{m}.py")):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        exports[m] = names
    bad = []
    for m in MODULES:
        for node in ast.walk(_tree(ROOT / f"{m}.py")):
            if isinstance(node, ast.ImportFrom) and node.module in exports:
                for a in node.names:
                    if a.name not in exports[node.module]:
                        bad.append(f"{m}.py: from {node.module} import {a.name}")
    assert not bad, bad


def test_help_tooltip_keys_all_defined():
    src = APP.read_text()
    used = set(re.findall(r'HELP\["([a-z_]+)"\]', src))
    keys = set(re.findall(r'^\s+"([a-z_]+)":', src, re.M))
    assert not (used - keys), f"HELP keys missing: {sorted(used - keys)}"


def test_no_duplicate_static_widget_keys():
    keys = re.findall(r'key="([^"]+)"', APP.read_text())
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"duplicate widget keys: {dups}"


def test_all_third_party_imports_are_installed():
    """Catches the missing-requirements class of failure (the plotly crash).
    Skipped in stub environments; in CI it runs against real installs."""
    if conftest.STUB_MODE:
        pytest.skip("dependency stubs active — run in CI with real installs")
    import sys
    local = set(MODULES)
    stdlib = set(sys.stdlib_module_names)
    third_party = set()
    for m in MODULES:
        for node in ast.walk(_tree(ROOT / f"{m}.py")):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            third_party.update(x for x in mods if x not in stdlib and x not in local)
    missing = []
    for pkg in sorted(third_party):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    assert not missing, f"imported but not installed (fix requirements.txt): {missing}"
