"""AI4LCC OCSGE label helpers (MultiSenGE 14-class + MultiSenNA class 15)."""

from __future__ import annotations #cleanly
from typing import Iterable

# MultiSenGE uses IDs 1–14. MultiSenNA adds coastal class 15 (Beaches, Sand).
AI4LCC_CLASS_ID_TO_NAME: dict[int, str] = {
    1: "Dense Built-Up",
    2: "Sparse Built-Up",
    3: "Specialized Built-Up Areas",
    4: "Specialized but Vegetative Areas",
    5: "Large Scale Networks",
    6: "Arable Lands",
    7: "Vineyards",
    8: "Orchards",
    9: "Grasslands",
    10: "Groves and Hedges",
    11: "Forests",
    12: "Open Spaces, Mineral",
    13: "Wetlands",
    14: "Water Surfaces",
    15: "Beaches, Sand",
}

AI4LCC_NAME_TO_ID: dict[str, int] = {v: k for k, v in AI4LCC_CLASS_ID_TO_NAME.items()}

URBAN_CLASS_IDS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
NATURAL_CLASS_IDS: frozenset[int] = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})

def parse_ai4lcc_label_ids(labels_field: str) -> list[int]:
    #Parsing labels ans storing labels in list
    if not labels_field or not str(labels_field).strip():
        return []
    return [int(x.strip()) for x in str(labels_field).split(";") if x.strip()]

def ai4lcc_name(class_id: int) -> str: #6: "Arable Lands"
    if class_id not in AI4LCC_CLASS_ID_TO_NAME:
        raise KeyError(f"Unknown AI4LCC class id: {class_id}")
    return AI4LCC_CLASS_ID_TO_NAME[class_id]

def ai4lcc_names_from_ids(class_ids: Iterable[int]) -> list[str]:
    #Return sorted unique class names present in a patch.
    seen: set[int] = set()
    names: list[str] = []
    for cid in sorted(class_ids):
        if cid in seen:
            continue
        seen.add(cid)
        names.append(ai4lcc_name(cid)) #groung truth label
    return names

def format_present_class_names(names: list[str]) -> str:
    return ", ".join(names) if names else "Unknown"

def dominant_ocsge_name(class_ids: Iterable[int]) -> str:
    """Deterministic dominant label: lowest present OCSGE class id."""
    ids = sorted(set(class_ids))
    if not ids:
        return "Unknown"
    return ai4lcc_name(ids[0])

#option for classify question (1–15 so MultiSenNA beaches are included)
def ai4lcc_classify_options_text() -> str:
    return ", ".join(AI4LCC_CLASS_ID_TO_NAME[i] for i in sorted(AI4LCC_CLASS_ID_TO_NAME))

def natural_class_names_from_ids(class_ids: Iterable[int]) -> list[str]:
    return [ai4lcc_name(cid) for cid in sorted(set(class_ids)) if cid in NATURAL_CLASS_IDS]

def urban_class_names_from_ids(class_ids: Iterable[int]) -> list[str]:
    return [ai4lcc_name(cid) for cid in sorted(set(class_ids)) if cid in URBAN_CLASS_IDS]
