"""
knowledge_graph.py
──────────────────
Layer 2: Visual Knowledge / Attributes & Relationships
Implements Visual Genome-style knowledge representation:
1. Object attribute extraction (colors, sizes, spatial zones, poses)
2. Pairwise object relationship graph (near, left of, right of, above, below, on, wearing)
3. Direct queries for visual relationships (e.g., 'car -> red', 'person -> near -> car')
"""

import math
from typing import Dict, List, Any, Optional, Tuple


# Semantic interaction predicates inspired by Visual Genome
WEARABLE_CLASSES = {"tie", "backpack", "umbrella", "handbag", "suitcase", "hat"}
VEHICLE_CLASSES = {"bicycle", "motorcycle", "car", "bus", "train", "truck", "boat", "airplane", "horse"}
SURFACE_CLASSES = {"dining table", "bed", "couch", "chair", "bench"}


class VisualKnowledgeGraph:
    """
    Builds and queries a Visual Genome-style knowledge graph from detected elements.
    """

    def __init__(self):
        pass

    def build_graph(self, detections: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
        """
        Builds attribute sets and relational triplets for all detected objects.
        """
        attributes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        triplets_text: List[str] = []

        # 1. Attribute Mapping
        for d in detections:
            rank = d.get("rank", d.get("id", 1))
            label = d.get("label", "object")
            color = d.get("dominant_color", "Unknown")
            prominence = d.get("prominence", "Midground")
            pos_desc = d.get("position", {}).get("location_desc", "center")

            attr_entry = {
                "rank": rank,
                "label": label,
                "color": color,
                "prominence": prominence,
                "position": pos_desc,
                "triplets": [
                    f"{label} #{rank} is {color.lower()}",
                    f"{label} #{rank} is in {prominence.lower()}",
                    f"{label} #{rank} is located at {pos_desc}"
                ]
            }
            attributes.append(attr_entry)
            triplets_text.extend(attr_entry["triplets"])

        # 2. Pairwise Relationship Extraction
        n = len(detections)
        diag = math.sqrt(img_w ** 2 + img_h ** 2) if (img_w > 0 and img_h > 0) else 1000.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                d1 = detections[i]
                d2 = detections[j]

                r1 = d1.get("rank", d1.get("id", i + 1))
                r2 = d2.get("rank", d2.get("id", j + 1))
                l1 = d1.get("label", "object")
                l2 = d2.get("label", "object")

                b1 = d1["bbox"]  # [x1, y1, x2, y2]
                b2 = d2["bbox"]

                c1_x = (b1[0] + b1[2]) / 2.0
                c1_y = (b1[1] + b1[3]) / 2.0
                c2_x = (b2[0] + b2[2]) / 2.0
                c2_y = (b2[1] + b2[3]) / 2.0

                dx = c1_x - c2_x
                dy = c1_y - c2_y
                dist = math.sqrt(dx ** 2 + dy ** 2)
                norm_dist = dist / diag

                # Check bounding box containment / overlap
                is_b1_inside_b2 = (b1[0] >= b2[0] - 10 and b1[1] >= b2[1] - 10 and
                                  b1[2] <= b2[2] + 10 and b1[3] <= b2[3] + 10)
                
                # Semantic Relations
                # A) Structural & Interaction Relations
                if "balcony" in l1 and "building" in l2:
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": "on facade of",
                        "object": f"{l2} #{r2}",
                        "summary": f"{l1} #{r1} is on the facade of {l2} #{r2}"
                    }
                    relationships.append(rel)
                    triplets_text.append(rel["summary"])
                elif "tire" in l1 and "bus" in l2:
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": "part of",
                        "object": f"{l2} #{r2}",
                        "summary": f"{l1} #{r1} is on {l2} #{r2}"
                    }
                    relationships.append(rel)
                    triplets_text.append(rel["summary"])
                elif l1 == "person" and "floor" in l2:
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": "standing/walking on",
                        "object": f"{l2} #{r2}",
                        "summary": f"{l1} #{r1} is walking on {l2} #{r2}"
                    }
                    relationships.append(rel)
                    triplets_text.append(rel["summary"])
                elif is_b1_inside_b2:
                    if l1 in WEARABLE_CLASSES and l2 == "person":
                        rel = {
                            "subject": f"{l2} #{r2}",
                            "predicate": "wearing",
                            "object": f"{l1} #{r1}",
                            "summary": f"{l2} #{r2} is wearing {l1} #{r1}"
                        }
                        relationships.append(rel)
                        triplets_text.append(rel["summary"])
                    elif l2 in SURFACE_CLASSES:
                        rel = {
                            "subject": f"{l1} #{r1}",
                            "predicate": "on",
                            "object": f"{l2} #{r2}",
                            "summary": f"{l1} #{r1} is on {l2} #{r2}"
                        }
                        relationships.append(rel)
                        triplets_text.append(rel["summary"])

                # B) Proximity (near / adjacent)
                if norm_dist < 0.28:
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": "near",
                        "object": f"{l2} #{r2}",
                        "distance_norm": round(norm_dist, 3),
                        "summary": f"{l1} #{r1} is near {l2} #{r2}"
                    }
                    relationships.append(rel)
                    triplets_text.append(rel["summary"])

                # C) Relative Spatial Orientation (left of / right of / above / below)
                if abs(dx) > abs(dy) * 1.2:
                    spatial_pred = "to the right of" if dx > 0 else "to the left of"
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": spatial_pred,
                        "object": f"{l2} #{r2}",
                        "summary": f"{l1} #{r1} is {spatial_pred} {l2} #{r2}"
                    }
                    relationships.append(rel)
                elif abs(dy) > abs(dx) * 1.2:
                    spatial_pred = "below" if dy > 0 else "above"
                    rel = {
                        "subject": f"{l1} #{r1}",
                        "predicate": spatial_pred,
                        "object": f"{l2} #{r2}",
                        "summary": f"{l1} #{r1} is {spatial_pred} {l2} #{r2}"
                    }
                    relationships.append(rel)

        return {
            "attributes": attributes,
            "relationships": relationships,
            "sample_triplets": triplets_text[:12]
        }
