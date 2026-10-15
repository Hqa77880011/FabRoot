import math
import re

import numpy as np
import torch
from torch import nn


SOURCES = ["image", "metrology", "alarm", "text"]
CAUSES = [
    "recipe drift",
    "chamber contamination",
    "sensor calibration error",
    "wafer alignment error",
    "gas-delivery deviation",
    "temperature-control deviation",
    "wafer-handling damage",
    "inspection-system artifact",
]
KEYS = ["wafer", "lot", "step", "tool", "chamber"]


def words(text):
    return re.findall(r"[a-z0-9_.-]+", str(text).lower())


def normalized_fact(fact):
    return tuple(
        " ".join(str(fact[k]).strip().lower().split())
        for k in ["subject", "relation", "value"]
    )


def verified(fact, record):
    return normalized_fact(fact) in {
        normalized_fact(v) for v in record.get("facts", [])
    }


class FabRoot(nn.Module):
    def __init__(self, vocabulary, entities, width=128, variant="full"):
        super().__init__()
        self.vocabulary, self.entities, self.width, self.variant = (
            vocabulary,
            entities,
            width,
            variant,
        )
        self.visual = nn.Linear(768, width)
        self.coordinate = nn.Linear(2, width)
        self.word = nn.Embedding(len(vocabulary), width)
        self.alarm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout=0.1, batch_first=True
            ),
            4,
        )
        self.text = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout=0.1, batch_first=True
            ),
            4,
        )
        self.metrology = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.field = nn.Embedding(64, width)
        self.kind = nn.Embedding(4, width)
        self.entity = nn.Embedding(len(entities), width)
        self.availability = nn.Embedding(16, width)
        self.time = nn.ModuleList([nn.Linear(16, width) for _ in SOURCES])
        self.reliability = nn.Linear(width + 5, 1)
        self.causes = nn.Parameter(torch.randn(8, width) * 0.02)
        self.graphs = nn.ModuleList(
            [nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(2)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.edge_bias = nn.Parameter(torch.zeros(4))
        self.support_token, self.support_cause = (
            nn.Linear(width, width),
            nn.Linear(width, width),
        )
        self.opposition_token, self.opposition_cause = (
            nn.Linear(width, width),
            nn.Linear(width, width),
        )
        self.ground = nn.Linear(width * 2, 1)
        self.concatenation = nn.Linear(width * 4, 8)

    def tokenize(self, text, device):
        ids = [self.vocabulary.get(w, 1) for w in words(text)[:64]] or [1]
        return torch.tensor(ids, device=device)

    def forward(self, event, dataset_root, mean, scale, time_shift=0.0):
        device = self.causes.device
        available = [
            any(r["type"] == source for r in event["records"]) for source in SOURCES
        ]
        bitmask = sum(int(v) << i for i, v in enumerate(available))
        context_ids = [
            self.entities.get(k + "=" + str(event["key"][k]), 1)
            for k in ["step", "tool", "chamber"]
        ]
        context = self.entity(torch.tensor(context_ids, device=device)).sum(0)
        tokens, information = [], []
        for record in event["records"]:
            source = SOURCES.index(record["type"])
            if record["type"] == "image":
                features = torch.tensor(
                    np.load(dataset_root / record["features"]), device=device
                )
                value = self.visual(features) + self.coordinate(
                    torch.tensor(
                        record["coordinate"], device=device, dtype=torch.float32
                    )
                )
            elif record["type"] == "metrology":
                valid = np.asarray(record["valid"], dtype=bool)
                fields = np.flatnonzero(valid)
                if len(fields) == 0:
                    continue
                numeric = (
                    np.asarray(record["values"], dtype=float)[valid] - mean[valid]
                ) / scale[valid]
                features = torch.tensor(
                    np.column_stack([numeric, np.ones(len(numeric))]),
                    device=device,
                    dtype=torch.float32,
                )
                value = self.metrology(features) + self.field(
                    torch.tensor(fields, device=device)
                )
            else:
                text = record.get("text", record.get("code", ""))
                ids = self.tokenize(text, device)
                sequence = self.word(ids)
                position = torch.arange(len(ids), device=device)[:, None]
                frequency = torch.exp(
                    torch.arange(0, self.width, 2, device=device)
                    * (-math.log(10000.0) / self.width)
                )
                encoding = torch.zeros_like(sequence)
                encoding[:, 0::2] = torch.sin(position * frequency)
                encoding[:, 1::2] = torch.cos(position * frequency)
                encoder = self.text if record["type"] == "text" else self.alarm
                value = encoder((sequence + encoding)[None])[0].mean(0, keepdim=True)
            delta = float(record["delta"]) + (time_shift if source else 0.0)
            frequencies = torch.exp(
                torch.arange(8, device=device) * (-math.log(10000.0) / 8)
            )
            time_encoding = torch.cat(
                [
                    torch.sin(delta * frequencies / 60),
                    torch.cos(delta * frequencies / 60),
                ]
            )
            aligned = (
                value + self.kind.weight[source] + self.availability.weight[bitmask]
            )
            if self.variant != "no_alignment":
                aligned = aligned + context + self.time[source](time_encoding)
            for vector in aligned:
                tokens.append(vector)
                information.append({"record": record, "source": source, "delta": delta})
        h0 = (
            torch.stack(tokens) if tokens else torch.empty(0, self.width, device=device)
        )
        quality = torch.tensor(
            [
                [
                    float(v["record"].get("qc", 1)),
                    float(v["record"].get("duplicate", 0)),
                    float(v["record"].get("entity_match", 1)),
                    float(v["record"].get("valid_fraction", 1)),
                    math.log1p(abs(v["delta"])),
                ]
                for v in information
            ],
            device=device,
        ).reshape(-1, 5)
        reliability = (
            0.05
            + 0.95 * torch.sigmoid(self.reliability(torch.cat([h0, quality], -1)))[:, 0]
        )
        if self.variant == "no_reliability":
            reliability = torch.ones_like(reliability)
        hidden = torch.cat([h0, self.causes])
        count = len(hidden)
        edge = torch.zeros(count, count, 4, device=device)
        if information:
            keys = [tuple(v["record"]["key"][key] for key in KEYS) for v in information]
            identifiers = {key: i for i, key in enumerate(dict.fromkeys(keys))}
            key_ids = torch.tensor([identifiers[key] for key in keys], device=device)
            entity_values = [str(v["record"].get("entity", "")) for v in information]
            entity_vocabulary = {"": 0}
            for value in entity_values:
                entity_vocabulary.setdefault(value, len(entity_vocabulary))
            entity_ids = torch.tensor(
                [entity_vocabulary[v] for v in entity_values], device=device
            )
            deltas = torch.tensor([v["delta"] for v in information], device=device)
            tags = sorted(
                {
                    tag
                    for v in information
                    for tag in v["record"].get("ontology_tags", [])
                }
            )
            tag_incidence = torch.tensor(
                [
                    [float(tag in v["record"].get("ontology_tags", [])) for tag in tags]
                    for v in information
                ],
                device=device,
            ).reshape(len(information), len(tags))
            length = len(information)
            edge[:length, :length, 0] = (key_ids[:, None] == key_ids[None]).float()
            edge[:length, :length, 1] = torch.exp(
                -(deltas[:, None] - deltas[None]).abs() / 600
            )
            edge[:length, :length, 2] = (
                (entity_ids[:, None] == entity_ids[None]) & (entity_ids[:, None] > 0)
            ).float()
            edge[:length, :length, 3] = (tag_incidence @ tag_incidence.T > 0).float()
        edge[-8:, :, 0], edge[:, -8:, 0] = 1.0, 1.0
        bias = edge @ self.edge_bias
        for graph, norm in zip(self.graphs, self.norms):
            message, _ = graph(
                hidden[None],
                hidden[None],
                hidden[None],
                attn_mask=bias,
                need_weights=False,
            )
            hidden = norm(hidden + message[0])
        evidence, causes = hidden[:-8], hidden[-8:]
        support = torch.sigmoid(
            self.support_token(evidence)
            @ self.support_cause(causes).T
            / math.sqrt(self.width)
        )
        opposition = torch.sigmoid(
            self.opposition_token(evidence)
            @ self.opposition_cause(causes).T
            / math.sqrt(self.width)
        )
        if self.variant == "no_opposition":
            opposition = torch.zeros_like(opposition)
        score = ((support - 0.6 * opposition) * reliability[:, None]).sum(0) / (
            reliability.sum() + 1e-7
        )
        signed_score = score
        if self.variant == "early_concat":
            pooled = [
                h0[[v["source"] == source for v in information]].mean(0)
                if any(v["source"] == source for v in information)
                else torch.zeros(self.width, device=device)
                for source in range(4)
            ]
            score = self.concatenation(torch.cat(pooled))
        return {
            "score": score,
            "signed_score": signed_score,
            "support": support,
            "opposition": opposition,
            "reliability": reliability,
            "hidden": evidence,
            "initial": h0,
            "information": information,
            "available": available,
        }

    def claim_score(self, representation, fact):
        ids = self.tokenize(" ".join(normalized_fact(fact)), representation.device)
        return self.ground(torch.cat([representation, self.word(ids).mean(0)]))[0]
