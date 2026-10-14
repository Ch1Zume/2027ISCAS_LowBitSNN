"""Small, auditable intervention on a copied frozen WTA runtime."""

ANCHOR = "            selected = selected.reshape_as(spike)\n"
RESET = """            # Once per presentation: reset all membranes in this sample's layer.
            # Keep this step's spikes, winner eligibility, traces and thresholds.
            # The detached Boolean event also cuts gradients through cleared state.
            mem = mem.masked_fill(has_winner.any(dim=1, keepdim=True), 0.0)
"""


def reset_source(source):
    if source.count(ANCHOR) != 1:
        raise ValueError("Frozen WTA source no longer matches the audited insertion point")
    return source.replace(ANCHOR, ANCHOR + RESET)
