"""Extract lexical mentions from requirement excerpts without inferring fit."""

import re


def extract_mentions(
    record: dict, items: list[dict], vocabulary: list[dict]
) -> list[dict]:
    """Locate configured terms within evidence-bearing requirement excerpts.

    Mentions are not independently required skills: the original context, scope,
    alternatives, qualifications, and optionality remain essential to interpretation.

    Args:
        record: Vacancy containing its complete parsed text.
        items: Requirement excerpts with original character offsets.
        vocabulary: Curated term labels, categories, and regex patterns.

    Returns:
        Mentions with exact surface text and offsets into the vacancy text.
    """
    mentions = []
    patterns = [
        (entry, [re.compile(pattern) for pattern in entry["patterns"]])
        for entry in vocabulary
    ]
    for item in items:
        start, end = item["start"], item["end"]
        if record["text"][start:end] != item["text"]:
            raise ValueError(f"Invalid requirement offsets for {record['id']}")
        for entry, regexes in patterns:
            matches = {}
            for regex in regexes:
                for match in regex.finditer(item["text"]):
                    if (
                        match.start() not in matches
                        or match.end() > matches[match.start()].end()
                    ):
                        matches[match.start()] = match
            for match in matches.values():
                mentions.append(
                    {
                        "label": entry["label"],
                        "category": entry["category"],
                        "matched_text": match.group(),
                        "start": start + match.start(),
                        "end": start + match.end(),
                        "context": item["text"],
                        "context_start": start,
                        "context_end": end,
                        "priority": item["priority"],
                        "scope": item["scope"],
                    }
                )
    return sorted(
        mentions,
        key=lambda mention: (mention["start"], mention["end"], mention["label"]),
    )
