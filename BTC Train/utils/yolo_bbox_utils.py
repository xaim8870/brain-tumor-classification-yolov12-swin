def adjust_yolo_boxes(boxes, crop_box, orig_w, orig_h):
    """
    Adjust YOLO boxes after cropping.

    boxes: list of (cls, xc, yc, w, h) in normalized YOLO format
    crop_box: (x1, y1, x2, y2) in pixel coords
    orig_w, orig_h: original image size

    Returns:
        adjusted_boxes in normalized YOLO format
    """
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    adjusted = []

    for cls, xc, yc, bw, bh in boxes:
        # Convert to pixel coords
        cx = xc * orig_w
        cy = yc * orig_h
        bw_px = bw * orig_w
        bh_px = bh * orig_h

        # Shift due to crop
        cx -= x1
        cy -= y1

        # Drop boxes outside crop
        if cx <= 0 or cy <= 0 or cx >= crop_w or cy >= crop_h:
            continue

        # Normalize to new crop size
        new_xc = cx / crop_w
        new_yc = cy / crop_h
        new_bw = bw_px / crop_w
        new_bh = bh_px / crop_h

        adjusted.append((cls, new_xc, new_yc, new_bw, new_bh))

    return adjusted
