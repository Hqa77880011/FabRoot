"""Finite manufacturing interventions with pixel, metrology and process-log observations."""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from PIL import Image


PARAMETERS = ["dose", "particles", "sensor_bias", "alignment", "flow", "temperature", "scratch", "stray_light"]


def wafer(parameters, rng):
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    dose, particles, _, alignment, flow, temperature, scratch, light = parameters
    stripes = np.sin((15 + dose) * (xx + alignment * 0.04))
    texture = 0.5 + 0.22 * stripes + 0.04 * flow * xx + 0.04 * temperature * (xx**2 + yy**2)
    spots = rng.random((64, 64)) < max(0, 0.015 + particles * 0.04)
    texture -= spots * 0.3
    texture -= (np.abs(yy - 0.2 * xx) < 0.02) * max(0, scratch) * 0.2
    texture += light * 0.07 * np.cos(30 * yy)
    texture = np.where(xx**2 + yy**2 < 0.95, texture, 0.1)
    texture += rng.normal(0, 0.025, texture.shape)
    return Image.fromarray(np.clip(np.repeat(texture[:, :, None], 3, axis=2) * 255, 0, 255).astype(np.uint8))


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rng, events = np.random.default_rng(seed), []
    # Calibrated sensor sensitivities are drawn independently of class/event IDs.
    sensitivity = np.random.default_rng(771).normal(0, 0.25, (64, 8))
    sensitivity[:8] += np.eye(8)
    beginning = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    serial = 0
    for split, repeats in [("train", 6), ("val", 2), ("test", 2)]:
        classes = np.tile(np.arange(8), repeats)
        rng.shuffle(classes)
        for cause in classes:
            physical = rng.normal(0, 0.08, 8)
            physical[int(cause)] += rng.uniform(0.6, 1.4)
            measured = sensitivity @ physical + rng.normal(0, 0.15, 64)
            anchor = beginning + dt.timedelta(minutes=serial * 3)
            key = dict(wafer=f"wafer-{serial}", lot=f"{split}-lot-{serial // 4}", step="pattern",
                       tool=f"tool-{rng.integers(3)}", chamber=f"chamber-{rng.integers(2)}")
            base = dict(key=key, time=anchor.isoformat(), qc=True,
                        support=[int(cause)], oppose=[], entity="process-monitor")
            path = f"images/wafer-{serial:03d}.png"
            wafer(physical, rng).save(output / path)
            records = [dict(base, id=f"{serial}-image", type="image", path=path, after_step=True,
                            severity=float(np.std(measured)), coordinate=[[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]),
                       dict(base, id=f"{serial}-meter", type="metrology", values=measured.tolist(),
                            valid=[True] * 64, field_names=[f"sensor_{i}" for i in range(64)])]
            reading = int(np.argmax(np.abs(measured[:8])))
            fact = dict(subject=f"sensor_{reading}", relation="measured", value=round(float(measured[reading]), 3))
            records.append(dict(base, id=f"{serial}-alarm", type="alarm", code=f"channel_{reading}_deviation",
                                facts=[fact], ontology_tags=["process-sensor"]))
            records.append(dict(base, id=f"{serial}-text", type="text",
                                text=f"Monitor channel {reading} returned {fact['value']} units.", facts=[fact]))
            # These are separate annotations; the model forward pass consumes only observations.
            for record in records:
                if record["type"] == "image" and int(cause) == 2:
                    record["support"], record["oppose"] = [], [2]
            readings = [physical[0] + physical[3], physical[0], physical[1], physical[2],
                        physical[4], physical[5], physical[6], physical[7]]
            inspections = [dict(action=i, outcome=int(abs(value + rng.normal(0, 0.2)) > 0.45))
                           for i, value in enumerate(readings)]
            events.append(dict(id=f"event-{serial}", split=split, label=int(cause), key=key,
                               anchor=anchor.isoformat(), records=records, inspections=inspections,
                               input_source="finite_manufacturing_interventions"))
            serial += 1
    (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(parameters=PARAMETERS,
        sensor_model="64 mixed physical measurements with independent noise",
        image_encoder="four RGB pixel patches; not CLIP",
        annotations="separate intervention/evidence supervision, not forward features"), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    generate(Path(args.output), args.seed)


if __name__ == "__main__":
    main()
