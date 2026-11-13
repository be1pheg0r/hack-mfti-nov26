"""Composable postprocessing rules for evidence-validated demo entities."""

from decimal import Decimal
from typing import Protocol


class EntityRule(Protocol):
    """Transform entities of a particular category, preserving unrelated facts."""

    category: str

    def apply(self, entities: list[dict]) -> list[dict]:
        """Return the processed entities."""
        ...


class SumExperience:
    """Sum reported experience figures without asserting non-overlapping periods."""

    category = "experience"

    def apply(self, entities: list[dict]) -> list[dict]:
        """Combine positive experience facts while retaining each source and value."""
        selected = [
            entity
            for entity in entities
            if entity["category"] == self.category and entity["polarity"] != "absent"
        ]
        if len(selected) < 2:
            return entities
        total = float(sum(Decimal(str(entity["value"])) for entity in selected))
        aggregate = {
            **selected[0],
            "value": total,
            "display_value": total,
            "context": (
                "Сумма указанных значений в годах: "
                + " + ".join(str(entity["value"]) for entity in selected)
                + ". Пересечения и дубли не проверены."
            ),
            "sources": selected,
            "derived": True,
        }
        selected_ids = {id(entity) for entity in selected}
        result = []
        for entity in entities:
            if id(entity) not in selected_ids:
                result.append(entity)
            elif entity is selected[0]:
                result.append(aggregate)
        return result


RULES: dict[str, type[EntityRule]] = {"sum_experience": SumExperience}


def apply_rules(entities: list[dict], rule_names: list[str]) -> list[dict]:
    """Apply configured rules in order, rejecting unknown rule names."""
    for name in rule_names:
        try:
            rule = RULES[name]()
        except KeyError as exc:
            raise ValueError(f"Unknown postprocessing rule: {name}") from exc
        entities = rule.apply(entities)
    return entities
