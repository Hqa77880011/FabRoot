import argparse
import csv
import datetime as dt
import itertools
import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from .method import CAUSES, KEYS, SOURCES, FabRoot, normalized_fact, verified, words


def timestamp(value):
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(
            "Event anchors and record times require explicit timezone offsets"
        )
    return parsed.timestamp()


def prepare(args):
    events = json.loads(Path(args.source).read_text(encoding="utf-8"))
    if len({e["id"] for e in events}) != len(events) or any(
        e["label"] not in range(8) for e in events
    ):
        raise ValueError(
            "Event IDs must be unique and labels must use the eight registered causes"
        )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    excluded, vocabulary, entities, lots, prepared = (
        [],
        {"<pad>": 0, "<unk>": 1},
        {"<pad>": 0, "<unk>": 1},
        {},
        [],
    )
    vision = processor = None
    for event in events:
        if event["split"] not in ["train", "val", "test"]:
            raise ValueError("split must be train, val or test")
        lot = event["key"]["lot"]
        if lot in lots and lots[lot] != event["split"]:
            raise ValueError("A production lot crosses partitions")
        lots[lot] = event["split"]
        anchor = timestamp(event["anchor"])
        accepted, alarms = [], set()
        for source in event["records"]:
            record = json.loads(json.dumps(source))
            if record["type"] not in SOURCES:
                raise ValueError("Unknown record source")
            matches = all(record["key"][k] == event["key"][k] for k in KEYS)
            if not matches and not args.allow_foreign:
                excluded.append(
                    {
                        "event": event["id"],
                        "record": record["id"],
                        "reason": "event_key",
                    }
                )
                continue
            record["entity_match"] = float(matches and record.get("entity_match", True))
            record["delta"] = timestamp(record["time"]) - anchor
            delta, kind = record["delta"], record["type"]
            admitted = (
                (kind == "image" and record.get("after_step", False))
                or (
                    kind == "metrology"
                    and abs(delta) <= 900
                    and record.get("qc", False)
                )
                or (kind == "alarm" and abs(delta) <= 600)
                or (kind == "text" and -3600 <= delta <= 7200)
            )
            if not admitted:
                excluded.append(
                    {
                        "event": event["id"],
                        "record": record["id"],
                        "reason": "time_or_quality",
                    }
                )
                continue
            if kind == "alarm":
                signature = record.get("retransmission_key", record["id"])
                if signature in alarms:
                    excluded.append(
                        {
                            "event": event["id"],
                            "record": record["id"],
                            "reason": "duplicate",
                        }
                    )
                    continue
                alarms.add(signature)
            if kind == "text":
                text = re.sub(
                    r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[redacted]", record["text"]
                )
                record["text"] = re.sub(r"\b\+?\d[\d ()-]{8,}\d\b", "[redacted]", text)
                record.pop("author", None)
            if kind == "metrology":
                if len(record["values"]) != 64 or len(record["valid"]) != 64:
                    raise ValueError("Metrology needs 64 values and validity bits")
                valid = np.asarray(record["valid"], dtype=bool)
                if not np.isfinite(
                    np.asarray(record["values"], dtype=float)[valid]
                ).all():
                    raise ValueError("Valid metrology fields must be finite")
                if not valid.any():
                    continue
                record["valid_fraction"] = float(valid.mean())
                names = record.get("field_names", [f"metrology_{j}" for j in range(64)])
                record["facts"] = [
                    {
                        "subject": names[j],
                        "relation": "measured",
                        "value": record["values"][j],
                    }
                    for j in range(64)
                    if valid[j]
                ]
            accepted.append(record)
        images = sorted(
            [r for r in accepted if r["type"] == "image"], key=lambda r: -r["severity"]
        )[:4]
        metrology = sorted(
            [r for r in accepted if r["type"] == "metrology"],
            key=lambda r: abs(r["delta"]),
        )[:1]
        selected = (
            images
            + metrology
            + sorted(
                [r for r in accepted if r["type"] in ["alarm", "text"]],
                key=lambda r: r["delta"],
            )
        )
        if len({record["id"] for record in selected}) != len(selected):
            raise ValueError("Retained evidence IDs must be unique within an event")
        for record in selected:
            if record["type"] == "image":
                if not args.clip_path:
                    raise ValueError(
                        "Image preparation requires a local CLIP ViT-B/16 model directory via --clip-path"
                    )
                if vision is None:
                    from transformers import CLIPImageProcessor, CLIPVisionModel

                    vision = CLIPVisionModel.from_pretrained(
                        args.clip_path, local_files_only=True
                    ).eval()
                    processor = CLIPImageProcessor.from_pretrained(
                        args.clip_path, local_files_only=True
                    )
                from PIL import Image

                image_path = Path(args.source).parent / record["path"]
                with Image.open(image_path) as image:
                    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
                with torch.no_grad():
                    feature = vision(**inputs).last_hidden_state[0].numpy()
                if feature.shape[1] != 768:
                    raise ValueError(
                        "The visual backbone must provide 768-wide ViT-B/16 features"
                    )
                relative = f"images/region_{sum(len(e['records']) for e in prepared)}_{len(accepted)}_{selected.index(record)}.npy"
                np.save(root / relative, feature.astype(np.float32))
                record["features"] = relative
                record.pop("path")
            if event["split"] == "train":
                text = record.get("text", record.get("code", ""))
                text += " " + " ".join(
                    " ".join(normalized_fact(f)) for f in record.get("facts", [])
                )
                for word in words(text):
                    if (
                        word not in vocabulary
                        and len(vocabulary) < args.vocabulary_size
                    ):
                        vocabulary[word] = len(vocabulary)
        if event["split"] == "train":
            for key in ["step", "tool", "chamber"]:
                entities.setdefault(key + "=" + str(event["key"][key]), len(entities))
        prepared.append(
            {
                **{k: v for k, v in event.items() if k != "records"},
                "records": selected,
                "anchor_seconds": anchor,
            }
        )
    for left, right in [("train", "val"), ("val", "test")]:
        a = [e["anchor_seconds"] for e in prepared if e["split"] == left]
        b = [e["anchor_seconds"] for e in prepared if e["split"] == right]
        if not a or not b or max(a) > min(b):
            raise ValueError(
                "Require nonempty chronological train/val/test event periods"
            )
    values = [[] for _ in range(64)]
    for event in prepared:
        if event["split"] == "train":
            for record in event["records"]:
                if record["type"] == "metrology":
                    for j, valid in enumerate(record["valid"]):
                        if valid:
                            values[j].append(record["values"][j])
    mean = [float(np.mean(v)) if v else 0.0 for v in values]
    scale = [max(float(np.std(v)), 1e-6) if v else 1.0 for v in values]
    package = {
        "events": prepared,
        "vocabulary": vocabulary,
        "entities": entities,
        "mean": mean,
        "scale": scale,
    }
    (root / "dataset.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    (root / "excluded.json").write_text(
        json.dumps(excluded, indent=2), encoding="utf-8"
    )


def record_scores(output):
    grouped = {}
    for j, information in enumerate(output["information"]):
        identifier = information["record"]["id"]
        entry = grouped.setdefault(
            identifier, {"record": information["record"], "indices": []}
        )
        entry["indices"].append(j)
    return grouped


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    package = json.loads(Path(args.data).read_text(encoding="utf-8"))
    train_events = [e for e in package["events"] if e["split"] == "train"]
    val_events = [e for e in package["events"] if e["split"] == "val"]
    if args.source_only:
        train_events = [
            {**e, "records": [r for r in e["records"] if r["type"] == args.source_only]}
            for e in train_events
        ]
        val_events = [
            {**e, "records": [r for r in e["records"] if r["type"] == args.source_only]}
            for e in val_events
        ]
    model = FabRoot(
        package["vocabulary"], package["entities"], args.width, args.variant
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    mean, scale = np.asarray(package["mean"]), np.asarray(package["scale"])
    root, out = Path(args.data).parent, Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best, patience, history = -1.0, 0, []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_events))
        losses = []
        for begin in range(0, len(order), args.batch):
            batch = [train_events[int(i)] for i in order[begin : begin + args.batch]]
            outputs = [model(event, root, mean, scale) for event in batch]
            logits = torch.stack([v["score"] for v in outputs]) / 0.1
            labels = torch.tensor([e["label"] for e in batch], device=args.device)
            ranking = nn.functional.cross_entropy(logits, labels)
            true = logits.gather(1, labels[:, None])[:, 0]
            competing = (
                logits.clone()
                .scatter(1, labels[:, None], -torch.inf)
                .topk(2, dim=1)
                .values
            )
            ranking += (
                torch.logsumexp(torch.cat([true[:, None], competing], 1), 1) - true
            ).mean()
            evidence_loss, grounding_loss, represented, ground_examples = (
                logits.sum() * 0,
                logits.sum() * 0,
                0,
                0,
            )
            for event, output in zip(batch, outputs):
                for entry in record_scores(output).values():
                    record, ids = entry["record"], entry["indices"]
                    if "support" not in record or "oppose" not in record:
                        raise ValueError(
                            "Training records need explicit signed-link annotations, including empty lists"
                        )
                    support = torch.tensor(
                        [float(k in record["support"]) for k in range(8)],
                        device=args.device,
                    )
                    oppose = torch.tensor(
                        [float(k in record["oppose"]) for k in range(8)],
                        device=args.device,
                    )
                    evidence_loss += nn.functional.binary_cross_entropy(
                        output["support"][ids].mean(0), support
                    )
                    if args.variant != "no_opposition":
                        evidence_loss += nn.functional.binary_cross_entropy(
                            output["opposition"][ids].mean(0), oppose
                        )
                    representation = output["hidden"][ids].mean(0)
                    for fact in record.get("facts", []):
                        positive = model.claim_score(representation, fact)
                        negative = model.claim_score(
                            representation,
                            {**fact, "value": str(fact["value"]) + " contradicted"},
                        )
                        grounding_loss += nn.functional.softplus(
                            -positive
                        ) + nn.functional.softplus(negative)
                        ground_examples += 2
                    represented += 1
            alignment = logits.sum() * 0
            pooled = [
                nn.functional.normalize(v["initial"].mean(0), dim=0)
                if len(v["initial"])
                else None
                for v in outputs
            ]
            for i, output in enumerate(outputs):
                if pooled[i] is None:
                    continue
                negatives = [
                    j
                    for j, event in enumerate(batch)
                    if j != i
                    and pooled[j] is not None
                    and (
                        event["key"]["lot"] != batch[i]["key"]["lot"]
                        or event["key"]["chamber"] != batch[i]["key"]["chamber"]
                    )
                ]
                if not negatives:
                    continue
                targets = torch.stack([pooled[i], *[pooled[j] for j in negatives]])
                for source in range(4):
                    ids = [
                        j
                        for j, v in enumerate(output["information"])
                        if v["source"] == source
                    ]
                    if ids:
                        query = nn.functional.normalize(
                            output["initial"][ids].mean(0), dim=0
                        )
                        alignment += -nn.functional.log_softmax(
                            query @ targets.T / 0.1, 0
                        )[0]
            loss = ranking + 0.2 * alignment / max(1, len(batch))
            if args.variant != "no_evidence_loss":
                loss += 0.5 * evidence_loss / max(1, represented)
            if args.variant != "no_grounding":
                loss += 0.1 * grounding_loss / max(1, ground_examples)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_scores = np.stack(
                [model(e, root, mean, scale)["score"].cpu().numpy() for e in val_events]
            )
        ranks = [
            int(np.flatnonzero(np.argsort(-score) == e["label"])[0]) + 1
            for score, e in zip(val_scores, val_events)
        ]
        ndcg = float(np.mean([1 / math.log2(r + 1) if r <= 3 else 0 for r in ranks]))
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ndcg3": ndcg}
        )
        if ndcg > best:
            best, patience = ndcg, 0
            torch.save(
                {
                    "state": model.state_dict(),
                    "width": args.width,
                    "variant": args.variant,
                    "vocabulary": package["vocabulary"],
                    "entities": package["entities"],
                    "source_only": args.source_only,
                },
                out / "model.pt",
            )
        else:
            patience += 1
        if patience >= 7:
            break
    (out / "history.json").write_text(json.dumps(history, indent=2))


def entropy(p):
    return float(-(p * np.log(p.clip(1e-12))).sum())


def action_models(events, catalog):
    models = []
    for a, action in enumerate(catalog["actions"]):
        count = np.ones((8, len(action["outcomes"])))
        for event in events:
            if event["split"] == "train":
                for inspection in event.get("inspections", []):
                    if inspection["action"] == a:
                        count[event["label"], inspection["outcome"]] += 1
        models.append(count / count.sum(1, keepdims=True))
    return models


def next_action(p, models, catalog):
    values = []
    for likelihood, action in zip(models, catalog["actions"]):
        probability = p @ likelihood
        expected = sum(
            probability[z] * entropy(p * likelihood[:, z] / max(probability[z], 1e-12))
            for z in range(len(probability))
        )
        values.append((entropy(p) - expected) / action["cost"])
    return int(np.argmax(values)), list(map(float, values))


def retrieve(output, cause, sign, limit):
    grouped = record_scores(output)
    candidates = []
    for identifier, entry in grouped.items():
        ids = entry["indices"]
        signed = output["reliability"][ids] * (
            output["support"][ids, cause] - 0.6 * output["opposition"][ids, cause]
        )
        value = float(signed.mean().detach()) * sign
        if value > 0:
            candidates.append((value, identifier, entry))
    candidates.sort(key=lambda v: -v[0])
    selected = []
    for source in SOURCES:
        best = next(
            (
                candidate
                for candidate in candidates
                if candidate[2]["record"]["type"] == source
            ),
            None,
        )
        if best:
            selected.append(best)
    selected += [v for v in candidates if v[1] not in {s[1] for s in selected}]
    return selected[:limit]


def evidence_ap(output, cause, sign):
    scores, targets = [], []
    for entry in record_scores(output).values():
        record, ids = entry["record"], entry["indices"]
        field = "support" if sign > 0 else "oppose"
        if field not in record:
            return None
        scores.append(
            float(
                (output["support"] if sign > 0 else output["opposition"])[
                    ids, cause
                ].mean()
            )
        )
        targets.append(int(cause in record[field]))
    return float(average_precision_score(targets, scores)) if any(targets) else None


def evaluate(args):
    package = json.loads(Path(args.data).read_text(encoding="utf-8"))
    saved = torch.load(args.model, map_location=args.device, weights_only=False)
    model = FabRoot(
        saved["vocabulary"], saved["entities"], saved["width"], saved["variant"]
    ).to(args.device)
    model.load_state_dict(saved["state"])
    model.eval()
    catalog = json.loads(Path(args.actions).read_text())
    if len(catalog["actions"]) != 8 or any(a["cost"] <= 0 for a in catalog["actions"]):
        raise ValueError("Supply eight inspections with positive registered costs")
    models = action_models(package["events"], catalog)
    mean, scale, root = (
        np.asarray(package["mean"]),
        np.asarray(package["scale"]),
        Path(args.data).parent,
    )
    validation = [e for e in package["events"] if e["split"] == "val"]
    source_only = args.source_only or saved.get("source_only")
    if source_only:
        validation = [
            {**e, "records": [r for r in e["records"] if r["type"] == source_only]}
            for e in validation
        ]
    with torch.no_grad():
        val_scores = np.stack(
            [model(e, root, mean, scale)["score"].cpu().numpy() for e in validation]
        )
    labels = np.array([e["label"] for e in validation])

    def calibration(log_temperature):
        scores = val_scores / np.exp(log_temperature)
        from scipy.special import logsumexp

        return float(
            np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels])
        )

    fit = minimize_scalar(calibration, bounds=(-8, 4), method="bounded")
    if not fit.success:
        raise RuntimeError("Temperature calibration failed: " + fit.message)
    temperature = float(np.exp(fit.x))
    tests = [e for e in package["events"] if e["split"] == "test"]
    drops = (
        list(itertools.combinations(SOURCES, args.missing_count))
        if args.missing_count
        else [tuple(filter(None, args.drop.split(",")))]
    )
    records = []
    proposals = (
        json.loads(Path(args.proposals).read_text(encoding="utf-8"))
        if args.proposals
        else {}
    )
    with torch.no_grad():
        for dropped in drops:
            for original in tests:
                event = {
                    **original,
                    "records": [
                        r for r in original["records"] if r["type"] not in dropped
                    ],
                }
                if source_only:
                    event["records"] = [
                        r for r in event["records"] if r["type"] == source_only
                    ]
                begin = time.perf_counter()
                output = model(event, root, mean, scale, args.time_shift_minutes * 60)
                p = (output["score"] / temperature).softmax(0).cpu().numpy()
                ranking = np.argsort(-p)
                first = int(ranking[0])
                positive = retrieve(output, first, 1, 12)
                negative = retrieve(output, first, -1, 4)
                margin = float(output["signed_score"][first])
                abstain = (
                    sum(output["available"]) < 2 or p[first] < 0.55 or margin < 0.08
                )
                action, utility = next_action(p, models, catalog)
                identifier = original["id"] + (
                    "__drop_" + "_".join(dropped) if dropped else ""
                )
                propositions = []
                if not abstain:
                    propositions.append(
                        {
                            "id": identifier + ".p0",
                            "type": "candidate",
                            "diagnostic": True,
                            "cause": CAUSES[first],
                            "evidence": [v[1] for v in positive],
                        }
                    )
                    generated = []
                    admitted = {v[1]: v[2] for v in positive + negative}
                    for kind, retrieved in [
                        ("support", positive),
                        ("counter_evidence", negative),
                    ]:
                        for _, evidence_id, entry in retrieved:
                            generated += [
                                {"type": kind, "evidence": evidence_id, "fact": fact}
                                for fact in entry["record"].get("facts", [])[:1]
                            ]
                    for proposal in proposals.get(original["id"], generated):
                        evidence_id, fact = proposal["evidence"], proposal["fact"]
                        entry = admitted.get(evidence_id)
                        allowed = (
                            entry is not None
                            and proposal["type"] in ["support", "counter_evidence"]
                            and all(k in fact for k in ["subject", "relation", "value"])
                        )
                        if allowed and not args.no_verifier:
                            representation = output["hidden"][entry["indices"]].mean(0)
                            allowed = (
                                verified(fact, entry["record"])
                                and float(
                                    model.claim_score(representation, fact).sigmoid()
                                )
                                >= 0.5
                            )
                        proposition = {
                            "id": identifier + ".p" + str(len(propositions)),
                            "type": proposal["type"],
                            "diagnostic": bool(allowed),
                            "evidence": [evidence_id] if entry else [],
                        }
                        proposition["fact"] = fact if allowed else None
                        proposition["text"] = (
                            " ".join(
                                str(fact[k]) for k in ["subject", "relation", "value"]
                            )
                            if allowed
                            else "Evidence is insufficient for this statement."
                        )
                        propositions.append(proposition)
                    propositions.append(
                        {
                            "id": identifier + ".p" + str(len(propositions)),
                            "type": "uncertainty",
                            "diagnostic": False,
                            "confidence": float(p[first]),
                            "signed_margin": margin,
                        }
                    )
                    propositions.append(
                        {
                            "id": identifier + ".p" + str(len(propositions)),
                            "type": "next_action",
                            "diagnostic": False,
                            "action": catalog["actions"][action]["name"],
                            "information_per_cost": utility[action],
                        }
                    )
                else:
                    propositions.append(
                        {
                            "id": identifier + ".p0",
                            "type": "uncertainty",
                            "diagnostic": False,
                            "text": "Insufficient evidence for report release; request manual escalation.",
                        }
                    )
                true_rank = int(np.flatnonzero(ranking == original["label"])[0]) + 1
                record = {
                    "id": identifier,
                    "event": original["id"],
                    "lot": original["key"]["lot"],
                    "dropped": list(dropped),
                    "label": original["label"],
                    "ranking": ranking[:3].tolist(),
                    "probability": p.tolist(),
                    "support": [v[1] for v in positive],
                    "counter_evidence": [v[1] for v in negative],
                    "propositions": propositions,
                    "abstain": bool(abstain),
                    "signed_margin": margin,
                    "action": None if abstain else action,
                    "action_utility": utility,
                    "ndcg3": 1 / math.log2(true_rank + 1) if true_rank <= 3 else 0.0,
                    "support_ap": evidence_ap(output, original["label"], 1),
                    "opposition_ap": evidence_ap(output, original["label"], -1),
                    "latency_ms": (time.perf_counter() - begin) * 1000,
                }
                if "exposing_actions" in original:
                    record["cwas"] = (
                        not abstain and action in original["exposing_actions"]
                    ) * (
                        1
                        - catalog["actions"][action]["cost"]
                        / (1 + max(a["cost"] for a in catalog["actions"]))
                    )
                records.append(record)
    released = [r for r in records if not r["abstain"]]
    result = {
        "ndcg3": float(np.mean([r["ndcg3"] for r in records])),
        "macro_f1": f1_score(
            [r["label"] for r in records],
            [r["ranking"][0] for r in records],
            labels=list(range(8)),
            average="macro",
            zero_division=0,
        ),
        "evidence_map": float(
            np.mean(
                [
                    r[k]
                    for r in records
                    for k in ["support_ap", "opposition_ap"]
                    if r[k] is not None
                ]
            )
        )
        if any(
            r[k] is not None for r in records for k in ["support_ap", "opposition_ap"]
        )
        else None,
        "release_coverage": len(released) / len(records),
        "released_ndcg3": float(np.mean([r["ndcg3"] for r in released]))
        if released
        else None,
        "all_event_released_ndcg3": float(
            np.mean([r["ndcg3"] if not r["abstain"] else 0 for r in records])
        ),
        "cwas": float(np.mean([r["cwas"] for r in records]))
        if all("cwas" in r for r in records)
        else None,
        "grounding_precision": None,
        "unsupported_statement_rate": None,
        "temperature": temperature,
        "latency_median_ms": float(np.median([r["latency_ms"] for r in records])),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reports.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    plt.hist([max(r["probability"]) for r in records], bins=15)
    plt.xlabel("Top cause probability")
    plt.ylabel("Event count")
    plt.tight_layout()
    plt.savefig(out / "confidence.png", dpi=180)
    plt.close()


def audit(args):
    reports = json.loads(Path(args.reports).read_text(encoding="utf-8"))
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    released = {
        p["id"] for event in reports for p in event["propositions"] if p["diagnostic"]
    }
    verdicts = {}
    for row in annotations:
        if (
            row["id"] not in released
            or row["id"] in verdicts
            or row["status"] not in ["supported", "unsupported", "unadjudicated"]
        ):
            raise ValueError(
                "Audit IDs must uniquely identify released diagnostic propositions with a valid status"
            )
        verdicts[row["id"]] = row["status"]
    counts = {
        status: sum(
            verdicts.get(identifier, "unadjudicated") == status
            for identifier in released
        )
        for status in ["supported", "unsupported", "unadjudicated"]
    }
    denominator = len(released)
    result = {
        "diagnostic_propositions": denominator,
        "counts": counts,
        "grounding_precision": counts["supported"] / denominator
        if denominator
        else None,
        "unsupported_statement_rate": counts["unsupported"] / denominator
        if denominator
        else None,
        "unadjudicated_rate": counts["unadjudicated"] / denominator
        if denominator
        else None,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))


def workflow(args):
    rows = list(csv.DictReader(open(args.source, encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    interfaces = sorted({r["interface"] for r in rows})
    participants = sorted({r["participant"] for r in rows})
    endpoints = ["diagnostic_minutes", "normalized_cost", "correct_action"]
    summary = {
        interface: {
            key: float(
                np.mean([float(r[key]) for r in rows if r["interface"] == interface])
            )
            for key in endpoints
        }
        for interface in interfaces
    }
    comparisons = []
    for other in interfaces:
        if other == args.reference:
            continue
        for endpoint in endpoints:
            differences = []
            for participant in participants:
                a = [
                    float(r[endpoint])
                    for r in rows
                    if r["participant"] == participant
                    and r["interface"] == args.reference
                ]
                b = [
                    float(r[endpoint])
                    for r in rows
                    if r["participant"] == participant and r["interface"] == other
                ]
                if not a or not b:
                    raise ValueError(
                        "Every participant needs both interfaces for a paired workflow comparison"
                    )
                differences.append(np.mean(b) - np.mean(a))
            from scipy.stats import wilcoxon

            differences = np.asarray(differences)
            bootstrap = np.mean(
                rng.choice(differences, (10000, len(differences)), replace=True), axis=1
            )
            p = float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
            comparisons.append(
                {
                    "interface": other,
                    "endpoint": endpoint,
                    "mean_difference": float(differences.mean()),
                    "bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "p": p,
                }
            )
    order = np.argsort([r["p"] for r in comparisons])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(
            adjusted, min(1.0, comparisons[int(index)]["p"] * (len(order) - rank))
        )
        comparisons[int(index)]["holm_p"] = adjusted
    Path(args.output).write_text(
        json.dumps(
            {"interfaces": summary, "participant_paired_comparisons": comparisons},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clip-path")
    p.add_argument("--allow-foreign", action="store_true")
    p.add_argument("--vocabulary-size", type=int, default=20000)
    p = sub.add_parser("train")
    for key in ["data", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--variant",
        choices=[
            "full",
            "early_concat",
            "no_alignment",
            "no_reliability",
            "no_opposition",
            "no_evidence_loss",
            "no_grounding",
        ],
        default="full",
    )
    for key, default in [("epochs", 50), ("batch", 8), ("width", 128), ("seed", 11)]:
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--device", default="cpu")
    p.add_argument("--source-only", choices=["image", "metrology"])
    p = sub.add_parser("evaluate")
    for key in ["data", "model", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--actions", default="actions.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--drop", default="")
    p.add_argument("--missing-count", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--time-shift-minutes", type=float, default=0.0)
    p.add_argument("--source-only", choices=["image", "metrology"])
    p.add_argument("--no-verifier", action="store_true")
    p.add_argument("--proposals")
    p = sub.add_parser("audit")
    for key in ["reports", "annotations", "output"]:
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("workflow")
    for key in ["source", "reference", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "train": train,
        "evaluate": evaluate,
        "audit": audit,
        "workflow": workflow,
    }[args.command](args)
