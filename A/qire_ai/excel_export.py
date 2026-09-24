from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def optional_xlsx_export(xlsx_path: Path, sheets: Dict[str, tuple[List[str], List[Dict[str, object]]]]) -> None:
    """Optional local xlsx export for users who install openpyxl.

    ChatGPT updates the official workbook separately. This helper is included so the
    project can also create a simple local workbook when run outside ChatGPT.
    """
    try:
        from openpyxl import Workbook
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is required for optional_xlsx_export") from exc

    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, (headers, rows) in sheets.items():
        ws = wb.create_sheet(sheet_name[:31])
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h, "") for h in headers])
    wb.save(xlsx_path)
