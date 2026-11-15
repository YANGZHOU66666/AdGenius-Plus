"""Stable text rendering shared by generation, annotation, and evaluation."""

from typing import Any


# These fields are historical, commercial, or media metadata rather than
# stable product facts for this text-only generation task.
_EXCLUDED_DETAIL_KEYS = {
    "best sellers rank",
    "date first available",
    "is discontinued by manufacturer",
    "manufacturer",
    "manufacturer part number",
    "item model number",
}


def _as_text(value: Any) -> str:
    """Render a source value as readable text without exposing JSON syntax."""
    if isinstance(value, list):
        return "; ".join(_as_text(item) for item in value if _as_text(item))
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {_as_text(item)}" for key, item in value.items()
            if _as_text(item)
        )
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [text for item in value if (text := _as_text(item))]
    text = _as_text(value)
    return [text] if text else []


def render_product_information(item: dict[str, Any]) -> str:
    product_info = item.get("product_info")
    if not isinstance(product_info, dict):
        raise ValueError("ProductItem.product_info must be an object")

    # Keep the ProductItem unchanged; this is only the model-input rendering.
    # Product IDs and commercial/media metadata do not help write the ad.
    sections: list[str] = ["Product information:"]

    title = _as_text(product_info.get("title"))
    if title:
        sections.extend(["", "Product name:", title])

    categories = _lines(product_info.get("categories"))
    if categories:
        sections.extend(["", "Categories:", *[f"- {value}" for value in categories]])

    features = _lines(product_info.get("features"))
    if features:
        sections.extend(["", "Features:", *[f"- {value}" for value in features]])

    descriptions = _lines(product_info.get("description"))
    if descriptions:
        sections.extend(["", "Description:", *descriptions])

    details = product_info.get("details")
    if isinstance(details, dict):
        detail_lines = []
        for key, value in details.items():
            if str(key).strip().casefold() in _EXCLUDED_DETAIL_KEYS:
                continue
            text = _as_text(value)
            if text:
                detail_lines.append(f"- {key}: {text}")
        if detail_lines:
            sections.extend(["", "Product details:", *detail_lines])

    return "\n".join(sections)
