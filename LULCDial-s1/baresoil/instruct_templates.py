"""EarthDial-style instruction templates for AI4LCC S1 LULC dialogue."""

from __future__ import annotations

from .taxonomy import (
    ai4lcc_classify_options_text,
    format_present_class_names,
    natural_class_names_from_ids,
)

LULC_TOKEN = "[baresoil]"
S1_VH_10_TOKEN = "[s1_vh_10]"
CLASSIFY_TOKEN = "[classify]"


def build_classify_qa(present_names: list[str]) -> tuple[str, str]:
    options = ai4lcc_classify_options_text()
    question = (
        f"{LULC_TOKEN} {S1_VH_10_TOKEN} {CLASSIFY_TOKEN} <image>\n"
        f"Classify the land cover in this image. Choose all that apply from: {options}."
    )
    answer = format_present_class_names(present_names)
    return question, answer


def build_dialogue_turns(label_ids: list[int], present_names: list[str]) -> list[dict[str, str]]:
    """Two-turn dialogue: classify, then ask about natural classes."""
    natural_names = natural_class_names_from_ids(label_ids)
    q1 = (
        f"{LULC_TOKEN} {S1_VH_10_TOKEN} <image>\n"
        "What land cover classes do you see in this image?"
    )
    a1 = format_present_class_names(present_names)
    q2 = "Which of these are natural or agricultural areas?"
    a2 = format_present_class_names(natural_names) if natural_names else "None"
    return [
        {"from": "human", "value": q1},
        {"from": "gpt", "value": a1},
        {"from": "human", "value": q2},
        {"from": "gpt", "value": a2},
    ]
