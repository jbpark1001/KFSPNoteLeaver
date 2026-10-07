"""Explicit primary and robustness sentence allocation rules."""
def primary_thirds(sentences):
    """Manuscript primary rule: equal floor-sized first two bins, remainder last."""
    n=len(sentences)
    if n<3:
        return (' '.join(sentences[:1]), '', ' '.join(sentences[1:]))
    width=n//3
    return tuple(' '.join(x) for x in (sentences[:width],sentences[width:2*width],sentences[2*width:]))
