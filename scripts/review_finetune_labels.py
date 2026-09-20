# Quick accept/correct/reject pass over the pre-labelled frames from
# sample_finetune_frames.py. Not part of the pipeline -- one-off tooling
# for the fine-tuning experiment, following the same "fast decision per
# item, never raw data entry" spirit as src/review.py.
#
# Controls:
#   a       accept the shown box (proposed, or your correction) as-is
#   s       skip -- no valid swimmer box for this frame, excluded from training
#   click twice (top-left, then bottom-right) to redraw the box, then press a
#   u       undo the current redraw and go back to the original proposal
#   q       save progress and quit -- resumes where you left off next time

import argparse
import json
import os

import cv2

STATE = {"points": [], "box": None, "scale": 1.0}


def on_mouse(event, x, y, flags, param):
    # Clicks arrive in the DISPLAY window's (scaled-down) pixel space, but
    # STATE["box"] is stored in the image's own full-resolution space
    # everywhere else (it started as the proposed box, already full-res) --
    # convert immediately so a redraw doesn't get scaled down twice.
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = STATE["scale"]
        STATE["points"].append((x / scale, y / scale))
        if len(STATE["points"]) == 2:
            (x1, y1), (x2, y2) = STATE["points"]
            STATE["box"] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            STATE["points"] = []


def draw(display, box, label, corrected):
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0) if corrected else (0, 255, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "a=accept  s=skip(no swimmer)  click TL then BR to redraw  u=undo  q=save+quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Review pre-labelled frames for the fine-tuning experiment.")
    parser.add_argument("--dir", default="finetune_data")
    parser.add_argument("--display-width", type=int, default=1280)
    args = parser.parse_args()

    manifest_path = os.path.join(args.dir, "manifest.json")
    images_dir = os.path.join(args.dir, "images")
    with open(manifest_path) as f:
        manifest = json.load(f)

    pending = [m for m in manifest if m["status"] == "pending"]
    done = len(manifest) - len(pending)
    print(f"{len(pending)} pending, {done} already reviewed.")
    if not pending:
        print("Nothing left to review.")
        return

    window = "Review fine-tuning labels"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    reviewed_this_session = 0
    try:
        for i, item in enumerate(pending):
            img_path = os.path.join(images_dir, item["image"])
            frame = cv2.imread(img_path)
            if frame is None:
                item["status"] = "skip"
                continue

            h, w = frame.shape[:2]
            scale = min(1.0, args.display_width / w)
            disp_w, disp_h = int(w * scale), int(h * scale)

            STATE["points"], STATE["box"], STATE["scale"] = [], item.get("proposed_box"), scale
            corrected = False
            label = f"{i + 1}/{len(pending)}  {item['image']}  source={item['source']}"

            while True:
                display = cv2.resize(frame, (disp_w, disp_h)).copy()
                shown_box = [c * scale for c in STATE["box"]] if STATE["box"] else None
                draw(display, shown_box, label, corrected)
                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF

                proposed = item.get("proposed_box")
                if STATE["box"] is not None and list(STATE["box"]) != (list(proposed) if proposed else None) and not corrected:
                    corrected = True  # a redraw just happened this frame

                if key == ord("a"):
                    # STATE["box"] is already full-resolution (see on_mouse) --
                    # save as-is, no scaling.
                    item["final_box"] = list(STATE["box"]) if STATE["box"] else None
                    item["status"] = "accepted_corrected" if corrected else "accepted"
                    reviewed_this_session += 1
                    break
                if key == ord("s"):
                    item["final_box"] = None
                    item["status"] = "skip"
                    reviewed_this_session += 1
                    break
                if key == ord("u"):
                    STATE["box"] = item.get("proposed_box")
                    STATE["points"] = []
                    corrected = False
                if key == ord("q"):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nStopping early -- progress saved.")
    finally:
        cv2.destroyAllWindows()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    remaining = sum(1 for m in manifest if m["status"] == "pending")
    print(f"Reviewed {reviewed_this_session} this session. {remaining} still pending.")


if __name__ == "__main__":
    main()
