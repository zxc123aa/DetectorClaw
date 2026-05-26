from __future__ import annotations

import json
from pathlib import Path


def load_stack_entries(stack_config_path: Path) -> list[dict]:
    payload = json.loads(Path(stack_config_path).read_text(encoding="utf-8"))
    materials = payload.get("materials", [])
    entries: list[dict] = []
    for material_index, material in enumerate(materials):
        rcf = material.get("rcf")
        if not rcf:
            continue
        entries.append(
            {
                "material_name": material.get("material_name"),
                "material_index": material_index,
                "thickness": material.get("thickness"),
                "thickness_type": material.get("thickness_type"),
                "rcf_id": rcf.get("rcf_id"),
                "table_id": rcf.get("table_ID"),
                "cutoff_energy": rcf.get("Cutoff_ene"),
            }
        )
    return entries
