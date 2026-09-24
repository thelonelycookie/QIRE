from __future__ import annotations


def run_external_agent(*args, **kwargs):
    """External-agent placeholder.

    v0.6 intentionally does not force API keys or network access. An external
    model can be connected later by writing a bounded adapter that returns
    ResearchProposal objects and never edits code or results directly.
    """
    raise RuntimeError("External AI agent is not configured. Use --ai-agent local/off or implement a bounded adapter.")
