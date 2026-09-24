from __future__ import annotations
from typing import Iterable, Tuple, List
from qire_ai.discovery.proposal_schema import ResearchProposal, validate_proposal


def validate_agent_proposals(proposals: Iterable[ResearchProposal]) -> Tuple[List[ResearchProposal], List[dict]]:
    ok, blocked = [], []
    for p in proposals:
        valid, reason = validate_proposal(p)
        if valid:
            ok.append(p)
        else:
            d = p.as_dict(); d["blocked_reason"] = reason; blocked.append(d)
    return ok, blocked
