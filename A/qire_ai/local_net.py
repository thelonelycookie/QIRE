from __future__ import annotations

"""Netz OHNE globale Mischschichten -- die Kontrolle, die A-003 gefehlt hat.

DAS PROBLEM
In allen bisherigen Aufbauten dieses Projekts steht vor und hinter der
komprimierten Mittelschicht eine dichte Schicht:

    784 -> [dense] -> 4096 -> [KOMPRIMIERT] -> 4096 -> [dense] -> 10

Globale Mischung ist damit immer verfuegbar, egal was die Mittelschicht tut.
A-003 wollte pruefen, ob die Blockdiagonale nur deshalb gut abschneidet, weil
ihre dichten Nachbarn das Mischen uebernehmen -- hat dafuer aber nur die
Mittelschichten gestapelt und die dichten Enden stehen lassen. A-003 zeigt
deshalb "mehr Blockschichten schaden nicht", nicht "das Netz braucht die
globale Mischung nicht". Das ist der Unterschied, den Monarch adressiert.

DIE KONTROLLE
Hier sind auch die Enden gruppenweise. Kein einziger gelernter Parameter
verbindet zwei Gruppen -- ausser der Mittelschicht, die gerade getestet wird:

    784 -> [16 Gruppen a 49 -> 256] -> 4096 -> [KOMPRIMIERT] -> 4096
        -> [16 Gruppen a 256 -> 10] -> Summe ueber Gruppen -> 10

Die Summe am Ende ist fest und ungelernt -- dieselbe Rolle wie global average
pooling in einem CNN. Sie kann keine gemeinsame Struktur zweier Gruppen
lernen; sie addiert nur fertige Beitraege. Ein Merkmal, das Information aus
zwei Gruppen verbindet, kann in diesem Netz also NUR in der Mittelschicht
entstehen.

Damit wird die Frage beantwortbar: Braucht das Netz globale Mischung, und
liefert die Mittelschicht sie?
"""

from typing import Callable

import torch
from torch import nn


class GroupedProjection(nn.Module):
    """Eingangsprojektion ohne Querverbindungen: Gruppe g sieht ausschliesslich
    ihren eigenen Abschnitt der Eingabe."""

    def __init__(self, in_features: int, out_features: int, groups: int, bias: bool = True):
        super().__init__()
        if in_features % groups or out_features % groups:
            raise ValueError("in_features und out_features muessen durch groups teilbar sein")
        self.groups = groups
        self.in_per = in_features // groups
        self.out_per = out_features // groups
        self.weight = nn.Parameter(torch.empty(groups, self.out_per, self.in_per))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = x.reshape(b, self.groups, self.in_per)
        h = torch.einsum("goi,bgi->bgo", self.weight, h)
        out = h.reshape(b, self.groups * self.out_per)
        return out + self.bias if self.bias is not None else out


class GroupedReadout(nn.Module):
    """Gruppenweise Klassifikation, danach feste Summe ueber die Gruppen.

    Die Summe ist bewusst ungelernt: sie fuegt keine lernbare Wechselwirkung
    zwischen Gruppen hinzu und liegt hinter der letzten Nichtlinearitaet."""

    def __init__(self, in_features: int, num_classes: int, groups: int):
        super().__init__()
        if in_features % groups:
            raise ValueError("in_features muss durch groups teilbar sein")
        self.groups = groups
        self.in_per = in_features // groups
        self.weight = nn.Parameter(torch.empty(groups, num_classes, self.in_per))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = x.reshape(b, self.groups, self.in_per)
        per_group = torch.einsum("gci,bgi->bgc", self.weight, h)
        return per_group.sum(dim=1) + self.bias


class MiddleNet(nn.Module):
    """Ein Netz, zwei Architekturvarianten, austauschbare Mittelschicht.

    arch = "dense_ends"  -- der bisherige Aufbau (dichte Enden)
    arch = "local_ends"  -- gruppenweise Enden, keine globale Mischung ausser
                            in der Mittelschicht
    """

    model_name = "MiddleNet"

    def __init__(self, input_dim: int, width: int, num_classes: int,
                 middle_factory: Callable[[], nn.Module], label: str, arch: str,
                 groups: int = 16, depth: int = 1):
        super().__init__()
        self.arch = arch
        if arch == "dense_ends":
            self.fc_in = nn.Linear(input_dim, width)
            self.fc_out = nn.Linear(width, num_classes)
        elif arch == "local_ends":
            self.fc_in = GroupedProjection(input_dim, width, groups)
            self.fc_out = GroupedReadout(width, num_classes, groups)
        else:
            raise ValueError(arch)
        self.middles = nn.ModuleList([middle_factory() for _ in range(depth)])
        self.method = f"{label}__{arch}"
        self.label = label

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Aktivierungen unmittelbar vor der Auslese -- fuer die Pruefung, ob
        Information Gruppengrenzen ueberquert hat."""
        h = torch.relu(self.fc_in(x))
        for layer in self.middles:
            h = torch.relu(layer(h))
        return h

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        return self.fc_out(h), {"active_model_parts": 1.0, "gate_entropy": 0.0,
                                "router_ops": 0, "expert_ops": 0}

    def middle_parameters(self) -> int:
        return sum(p.numel() for m in self.middles for p in m.parameters())

    def end_parameters(self) -> int:
        return sum(p.numel() for p in self.fc_in.parameters()) + \
               sum(p.numel() for p in self.fc_out.parameters())


def pad_input_to_groups(dim: int, groups: int) -> int:
    """Kleinste durch `groups` teilbare Zahl >= dim (MNIST: 784 = 16*49, passt)."""
    return dim if dim % groups == 0 else dim + (groups - dim % groups)
