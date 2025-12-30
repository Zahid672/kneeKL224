from PIL import Image, ImageDraw, ImageFont, ImageOps

# -----------------------------
# CONFIG
# -----------------------------
IN_PATH = "9453626L.png"  # Ensure this file exists!
OUT_PATH = "knee_yolo_pipeline.png"
BASE_SIZE = 256
BORDER = 4
LABEL_HEIGHT = 28
GAP = 20

# -----------------------------
# LOAD & RESIZE
# -----------------------------
img = Image.open(IN_PATH).convert("RGB")
img_resized = img.resize((BASE_SIZE, BASE_SIZE), Image.LANCZOS)

# Panel 1: original
panel1 = ImageOps.expand(img_resized, border=BORDER, fill="black")

# Panel 2: dummy YOLO ROI box
panel2 = img_resized.copy()
draw_box = ImageDraw.Draw(panel2)
w, h = panel2.size
box = (int(w * 0.18), int(h * 0.25), int(w * 0.82), int(h * 0.80))
draw_box.rectangle(box, outline="red", width=4)
panel2 = ImageOps.expand(panel2, border=BORDER, fill="black")

# Panel 3: auto-contrast preprocessing
panel3 = ImageOps.autocontrast(img_resized)
panel3 = ImageOps.expand(panel3, border=BORDER, fill="black")

# -----------------------------
# LABELING FUNCTION (fixed)
# -----------------------------
def add_label(panel, text):
    w, h = panel.size
    new_img = Image.new("RGB", (w, h + LABEL_HEIGHT), "white")
    new_img.paste(panel, (0, 0))

    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(((w - tw) // 2, h + (LABEL_HEIGHT - th) // 2), text, fill="black", font=font)
    return new_img

panel1_l = add_label(panel1, "Original knee crop")
panel2_l = add_label(panel2, "YOLO-like ROI detection")
panel3_l = add_label(panel3, "Preprocessed 224×224")

# -----------------------------
# COMBINE PANELS
# -----------------------------
total_width = panel1_l.width + panel2_l.width + panel3_l.width + 2 * GAP
max_height = max(panel1_l.height, panel2_l.height, panel3_l.height)

combined = Image.new("RGB", (total_width, max_height), "white")

x = 0
combined.paste(panel1_l, (x, 0))
x += panel1_l.width + GAP
combined.paste(panel2_l, (x, 0))
x += panel2_l.width + GAP
combined.paste(panel3_l, (x, 0))

combined.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
