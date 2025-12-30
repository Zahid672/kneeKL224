from PIL import Image, ImageDraw

# -------------------------
# CONFIG
# -------------------------
IMAGE_PATH = "E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\9025994L.png"
OUTPUT_PATH = "E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\image_with_14x14_grid.png"

IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = IMG_SIZE // PATCH_SIZE  # 14

# -------------------------
# LOAD & RESIZE IMAGE
# -------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))

draw = ImageDraw.Draw(img)

# -------------------------
# DRAW GRID LINES
# -------------------------
for i in range(1, NUM_PATCHES):
    x = i * PATCH_SIZE
    y = i * PATCH_SIZE

    # vertical line
    draw.line([(x, 0), (x, IMG_SIZE)], fill="white", width=1)

    # horizontal line
    draw.line([(0, y), (IMG_SIZE, y)], fill="white", width=1)

# -------------------------
# SAVE RESULT
# -------------------------
img.save(OUTPUT_PATH)

print("Saved grid image to:", OUTPUT_PATH)
