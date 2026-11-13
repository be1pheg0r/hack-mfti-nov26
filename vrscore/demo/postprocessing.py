"""Composable postprocessing rules for evidence-validated demo entities."""

from decimal import Decimal
from typing import Protocol


class EntityRule(Protocol):
    """Transform entities of specific categories, preserving unrelated facts."""

    categories: frozenset[str]

    def apply(self, entities: list[dict]) -> list[dict]:
        """Return the processed entities."""
        ...


class SumExperience:
    """Sum reported experience figures without asserting non-overlapping periods."""

    categories = frozenset({"experience"})

    def apply(self, entities: list[dict]) -> list[dict]:
        """Combine positive experience facts while retaining each source and value."""
        selected = [
            entity
            for entity in entities
            if entity["category"] in self.categories and entity["polarity"] != "absent"
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


class LocationMobilityByFormat:
    """Apply the vacancy work format's priority to location and mobility."""

    categories = frozenset({"location", "mobility"})

    def apply(self, entities: list[dict]) -> list[dict]:
        """Set priority without changing other fields or the input entities."""
        formats = {
            entity["value"]
            for entity in entities
            if entity["category"] == "work_format"
        }
        priority = (
            None
            if "remote" in formats
            else ("good2have" if "hybrid" in formats else "strict")
        )
        return [
            {
                **entity,
                "base_importance": entity.get(
                    "base_importance", entity.get("importance")
                ),
                "importance": priority,
            }
            if entity["category"] in self.categories
            else entity
            for entity in entities
        ]


RULES: dict[str, type[EntityRule]] = {
    "sum_experience": SumExperience,
    "location_mobility_by_format": LocationMobilityByFormat,
}


def apply_rules(entities: list[dict], rule_names: list[str]) -> list[dict]:
    """Apply configured rules in order, rejecting unknown rule names."""
    for name in rule_names:
        try:
            rule = RULES[name]()
        except KeyError as exc:
            raise ValueError(f"Unknown postprocessing rule: {name}") from exc
        entities = rule.apply(entities)
    return entities
