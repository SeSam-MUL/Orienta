"""Raises while being imported — what 'fails on activation' actually means
for an add-on whose only entry point is a named callable."""
raise RuntimeError("this add-on explodes on import")


def analyse(context):                       # pragma: no cover - never reached
    return []
