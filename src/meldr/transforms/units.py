"""Unit conversion table: length, area, volume, mass, duration (linear) and temperature."""

from __future__ import annotations

from meldr.core.errors import SpecValidationError

# unit -> (dimension, factor to the dimension's base unit)
_LINEAR: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    "yd": ("length", 0.9144),
    "mi": ("length", 1609.344),
    "m2": ("area", 1.0),
    "km2": ("area", 1e6),
    "ft2": ("area", 0.09290304),
    "ha": ("area", 10_000.0),
    "acre": ("area", 4046.8564224),
    "m3": ("volume", 1.0),
    "l": ("volume", 0.001),
    "ml": ("volume", 1e-6),
    "ft3": ("volume", 0.028316846592),
    "gal": ("volume", 0.003785411784),
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1e-6),
    "t": ("mass", 1000.0),
    "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "s": ("duration", 1.0),
    "ms": ("duration", 0.001),
    "min": ("duration", 60.0),
    "h": ("duration", 3600.0),
    "d": ("duration", 86_400.0),
}

_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "feet": "ft",
    "foot": "ft",
    "inch": "in",
    "inches": "in",
    "kilometer": "km",
    "mile": "mi",
    "miles": "mi",
    "sqft": "ft2",
    "sqm": "m2",
    "liter": "l",
    "litre": "l",
    "liters": "l",
    "gallon": "gal",
    "gallons": "gal",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "pound": "lb",
    "pounds": "lb",
    "ounce": "oz",
    "tonne": "t",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "day": "d",
    "days": "d",
    "c": "C",
    "celsius": "C",
    "f": "F",
    "fahrenheit": "F",
    "k": "K",
    "kelvin": "K",
}

TEMPERATURE = {"C", "F", "K"}
CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "INR", "CHF", "CNY"}


def canonical_unit(unit: str) -> str:
    u = unit.strip()
    if u in TEMPERATURE or u in _LINEAR:
        return u
    low = u.casefold()
    return _ALIASES.get(low, low)


def linear_factor(from_unit: str, to_unit: str) -> float:
    """Multiplier that converts `from_unit` to `to_unit` (same dimension, not temperature)."""
    f, t = canonical_unit(from_unit), canonical_unit(to_unit)
    if f.upper() in CURRENCIES or t.upper() in CURRENCIES:
        raise SpecValidationError(
            "Currency conversion is out of scope because it uses exchange rates. Leave the field "
            "for human review."
        )
    if f not in _LINEAR or t not in _LINEAR:
        unknown = [u for u in (from_unit, to_unit) if canonical_unit(u) not in _LINEAR]
        raise SpecValidationError(f"Unknown unit(s) {unknown}. Known: {sorted(_LINEAR)} + C/F/K.")
    (df, ff), (dt_, tf) = _LINEAR[f], _LINEAR[t]
    if df != dt_:
        raise SpecValidationError(f"Cannot convert {df} ({from_unit}) to {dt_} ({to_unit}).")
    return ff / tf


def is_known(unit: str) -> bool:
    u = canonical_unit(unit)
    return u in _LINEAR or u in TEMPERATURE
