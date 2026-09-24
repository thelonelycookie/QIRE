from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import SANDBOX_DIR

CANDIDATE_DIR = SANDBOX_DIR / "candidate_models"
REPORT_PATH = SANDBOX_DIR / "sandbox_mutation_report.jsonl"

FORBIDDEN_AST = (
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
)
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "subprocess", "os", "shutil", "pathlib"}

TEMPLATE = '''from __future__ import annotations
import torch
from torch import nn

class CandidateModel(nn.Module):
    model_name = "SandboxCandidateMLP"
    method = "sandbox_mutation_{variant}"
    def __init__(self, input_dim:int, hidden_dim:int, num_classes:int):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=max(2, int(hidden_dim*{hidden_factor}))
        self.num_classes=num_classes
        self.net=nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.{activation}(),
            nn.Linear(self.hidden_dim, num_classes),
        )
    def forward(self, x):
        logits=self.net(x)
        return logits, {{"active_model_parts": 1.0, "gate_entropy": 0.0}}
    def estimated_ops_per_sample(self)->int:
        return int(2*(self.input_dim*self.hidden_dim+self.hidden_dim*self.num_classes))
'''


def _validate_candidate_source(source: str) -> Dict[str, object]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"ok": False, "reason": f"syntax: {exc}"}
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_AST):
            return {"ok": False, "reason": f"forbidden ast node: {type(node).__name__}"}
        if isinstance(node, ast.Import):
            allowed = all(alias.name == "torch" for alias in node.names)
            if not allowed:
                return {"ok": False, "reason": "only import torch is allowed"}
        if isinstance(node, ast.ImportFrom):
            if node.module not in {"torch", "torch.nn", "__future__"}:
                return {"ok": False, "reason": "only torch imports are allowed"}
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_NAMES:
                return {"ok": False, "reason": f"forbidden call: {fn.id}"}
            if isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_NAMES:
                return {"ok": False, "reason": f"forbidden attr call: {fn.attr}"}
    if "class CandidateModel" not in source:
        return {"ok": False, "reason": "CandidateModel class missing"}
    return {"ok": True, "reason": "validated"}


def generate_sandbox_candidates(max_candidates: int = 6) -> List[Path]:
    """Generate candidate model files only inside results/sandbox.

    This is intentionally conservative. It does not modify project source code,
    does not execute shell commands from generated code, and does not import the
    generated file into the main process.
    """
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    variants = []
    for idx, (factor, activation) in enumerate([(0.25, "ReLU"), (0.5, "ReLU"), (0.75, "Tanh"), (1.0, "GELU"), (1.25, "ReLU"), (1.5, "Tanh")], start=1):
        source = TEMPLATE.format(variant=f"v{idx}", hidden_factor=factor, activation=activation)
        validation = _validate_candidate_source(source)
        path = CANDIDATE_DIR / f"candidate_model_v{idx}.py"
        report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "path": str(path), "validation": validation, "hidden_factor": factor, "activation": activation}
        if validation["ok"]:
            path.write_text(source, encoding="utf-8")
            variants.append(path)
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
        if len(variants) >= max_candidates:
            break
    return variants


def smoke_test_candidate(path: Path, timeout_sec: int = 8) -> Dict[str, object]:
    """Run a candidate import in a subprocess with timeout. Main code remains unchanged."""
    path = path.resolve()
    if CANDIDATE_DIR.resolve() not in path.parents:
        return {"ok": False, "path": str(path), "reason": "outside sandbox"}
    code = f'''
import importlib.util, torch
spec = importlib.util.spec_from_file_location("candidate", r"{path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m = mod.CandidateModel(13, 16, 3)
x = torch.randn(4, 13)
y, aux = m(x)
assert y.shape == (4, 3)
assert isinstance(aux, dict)
print("OK", m.estimated_ops_per_sample())
'''
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout_sec)
        return {"ok": res.returncode == 0, "path": str(path), "stdout": res.stdout.strip(), "stderr": res.stderr.strip(), "returncode": res.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "path": str(path), "reason": "timeout"}


def run_sandbox_code_mutation(max_candidates: int = 6, smoke_test: bool = True) -> Dict[str, object]:
    candidates = generate_sandbox_candidates(max_candidates=max_candidates)
    reports = []
    if smoke_test:
        for path in candidates:
            reports.append(smoke_test_candidate(path))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "candidate_paths": [str(p) for p in candidates],
        "smoke_tests": reports,
        "safety": "Sandbox-only. No project source files are modified. Human review required before using candidates in experiments.",
    }
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True) + "\n")
    return summary
