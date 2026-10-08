"""Readable person overlays; detection coordinates stay in source-image space."""
import colorsys

import cv2
import numpy as np

PANEL_WIDTH = 420
FONT = cv2.FONT_HERSHEY_SIMPLEX
REQUIREMENTS = ("mask", "gloves", "hair_cover", "apron")
CLASS_COLORS = {
    name: (118, 211, 111)
    for name in ("mask", "glove", "hairnet", "apron")
}
CLASS_COLORS.update({
    name: (115, 126, 255)
    for name in ("no_mask", "incorrect_mask", "no_glove", "no_hairnet", "no_apron")
})


def detected_class_rows(person):
    """Use current detections, with support for previously saved confirmed results."""
    if "detected_classes" in person:
        candidates = person["detected_classes"]
    else:
        # Older saved sessions contain only temporally confirmed evidence.
        candidates = [
            {"class_name": item.get("evidence_type"),
             "confidence": item.get("confidence", 0)}
            for key in REQUIREMENTS
            for item in [person.get(key, {})]
            if item.get("state") in {"compliant", "violation"}
        ]
    return [
        (item["class_name"], item["confidence"])
        for item in candidates
        if item.get("class_name") in CLASS_COLORS
    ]

def annotation_size(width, height, show_labels=True):
    # Enlarge the presentation, not the inference image. Even sizes suit H.264.
    out_height = max(720, min(1080, int(height)))
    out_height += out_height % 2
    out_width = max(2, round(width * out_height / height / 2) * 2)
    return out_width + (PANEL_WIDTH if show_labels else 0), out_height


def person_color(track_id):
    hue = (int(track_id) * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return tuple(round(channel * 255) for channel in (blue, green, red))


def _text(image, text, xy, color=(235, 238, 242), scale=0.48, thickness=1):
    cv2.putText(image, text, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def annotate_people(frame, persons, detections, *, show_boxes=True,
                    show_labels=True, show_confidence=True, show_raw_ppe=False):
    height, width = frame.shape[:2]
    out_width, out_height = annotation_size(width, height, show_labels)
    scene_width = out_width - (PANEL_WIDTH if show_labels else 0)
    scene = cv2.resize(frame, (scene_width, out_height), interpolation=cv2.INTER_LINEAR)
    sx, sy = scene_width / width, out_height / height
    people = sorted(persons, key=lambda p: p["track_id"])

    def scaled_box(bbox):
        x1, y1, x2, y2 = bbox
        return (max(0, min(scene_width - 1, round(x1 * sx))),
                max(0, min(out_height - 1, round(y1 * sy))),
                max(0, min(scene_width - 1, round(x2 * sx))),
                max(0, min(out_height - 1, round(y2 * sy))))

    if show_raw_ppe and show_boxes:
        for detection in detections:
            x1, y1, x2, y2 = scaled_box(detection["bbox"])
            cv2.rectangle(scene, (x1, y1), (x2, y2), (155, 155, 155), 1)

    occupied = []
    for person in people:
        x1, y1, x2, y2 = scaled_box(person["bbox"])
        color = person_color(person["track_id"])
        if show_boxes:
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
        if not show_labels:
            continue
        label = person["staff_label"]
        text_width = cv2.getTextSize(label, FONT, 0.5, 1)[0][0]
        badge_width, badge_height = min(scene_width, text_width + 14), 26
        candidates = [(x1, y1 - badge_height - 3), (x1, y2 + 3),
                      (x2 + 4, y1), (x1 - badge_width - 4, y1)]
        # Scan free space if labels near overlapping people would collide.
        candidates.extend((x1, y) for y in range(4, out_height - badge_height, 30))
        choices = []
        for left, top in candidates:
            left = max(0, min(scene_width - badge_width, left))
            top = max(0, min(out_height - badge_height, top))
            rect = (left, top, left + badge_width, top + badge_height)
            overlap = sum(max(0, min(rect[2], r[2]) - max(rect[0], r[0])) *
                          max(0, min(rect[3], r[3]) - max(rect[1], r[1]))
                          for r in occupied)
            choices.append((overlap, rect))
            if overlap == 0:
                break
        _, rect = min(choices, key=lambda item: item[0])
        occupied.append(rect)
        left, top, right, bottom = rect
        anchor = (max(x1, min(x2, (left + right) // 2)),
                  max(y1, min(y2, (top + bottom) // 2)))
        cv2.line(scene, anchor, ((left + right) // 2, (top + bottom) // 2), color, 1)
        cv2.rectangle(scene, (left, top), (right, bottom), color, -1)
        _text(scene, label, (left + 7, top + 18), (18, 22, 29), 0.5)

    if not show_labels:
        return scene

    # Render every visible person. Keep encoded dimensions fixed as people enter.
    panel_height = max(out_height, 84 + len(people) * 128)
    panel = np.full((panel_height, PANEL_WIDTH, 3), (27, 23, 20), dtype=np.uint8)
    _text(panel, "MODEL CLASSES", (16, 28), scale=0.65, thickness=2)
    _text(panel, "Only detected classes are listed", (16, 51), scale=0.45)
    if not people:
        _text(panel, "No confirmed person tracks", (16, 98))
    for index, person in enumerate(people):
        top = 68 + index * 128
        color = person_color(person["track_id"])
        cv2.rectangle(panel, (12, top), (PANEL_WIDTH - 12, top + 118), (43, 37, 32), -1)
        cv2.rectangle(panel, (12, top), (16, top + 118), color, -1)
        x1, y1, x2, y2 = person["bbox"]
        crop = frame[max(0, int(y1)):min(height, int(y2)),
                     max(0, int(x1)):min(width, int(x2))]
        if crop.size:
            ratio = min(62 / crop.shape[1], 92 / crop.shape[0])
            thumb = cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)),
                                     max(1, round(crop.shape[0] * ratio))))
            th, tw = thumb.shape[:2]
            panel[top + 12:top + 12 + th, 24:24 + tw] = thumb
        title = person["staff_label"]
        if show_confidence:
            title += "  person {:.0%}".format(person.get("person_confidence", 0))
        _text(panel, title, (98, top + 23), color, 0.5, 1)
        for row, (class_name, confidence) in enumerate(detected_class_rows(person)):
            text = class_name
            if show_confidence:
                text += f"  {confidence:.0%}"
            _text(panel, text, (98, top + 47 + row * 20),
                  CLASS_COLORS[class_name], 0.50)
    if panel_height != out_height:
        panel = cv2.resize(panel, (PANEL_WIDTH, out_height), interpolation=cv2.INTER_AREA)
    return np.concatenate((scene, panel), axis=1)
