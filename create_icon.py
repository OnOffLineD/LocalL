import math
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

# 1. 네온 글로우
glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow)
glow_draw.ellipse([80, 80, 432, 432], fill=(56, 189, 248, 45))
glow_draw.ellipse([140, 140, 372, 372], fill=(99, 102, 241, 35))
glow = glow.filter(ImageFilter.GaussianBlur(radius=40))
img.paste(glow, (0, 0), glow)

# 2. 스퀴클 플레이트
pad = 36
radius = 104
box = [pad, pad, SIZE - pad, SIZE - pad]

plate = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
p_draw = ImageDraw.Draw(plate)

grad = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
g_draw = ImageDraw.Draw(grad)
for y in range(pad, SIZE - pad):
    ratio = (y - pad) / (SIZE - 2 * pad)
    r = int(24 * (1 - ratio) + 10 * ratio)
    g = int(32 * (1 - ratio) + 14 * ratio)
    b = int(50 * (1 - ratio) + 24 * ratio)
    g_draw.line([(pad, y), (SIZE - pad, y)], fill=(r, g, b, 255))

plate_mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(plate_mask).rounded_rectangle(box, radius=radius, fill=255)
plate.paste(grad, (0, 0), plate_mask)

p_draw.rounded_rectangle(box, radius=radius, outline=(56, 189, 248, 140), width=4)

# 3. 돋보기 렌즈
cx, cy = 224, 224
cr = 92

lens_glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
lg_draw = ImageDraw.Draw(lens_glow)
lg_draw.ellipse([cx - cr + 4, cy - cr + 4, cx + cr - 4, cy + cr - 4], fill=(56, 189, 248, 30))
lens_glow = lens_glow.filter(ImageFilter.GaussianBlur(radius=16))
plate.paste(lens_glow, (0, 0), lens_glow)

p_draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=(56, 189, 248, 255), width=18)

# 돋보기 손잡이
hx1, hy1 = int(cx + cr * 0.707 + 6), int(cy + cr * 0.707 + 6)
hx2, hy2 = hx1 + 84, hy1 + 84
p_draw.line([(hx1, hy1), (hx2, hy2)], fill=(56, 189, 248, 255), width=20)
p_draw.ellipse([hx2 - 10, hy2 - 10, hx2 + 10, hy2 + 10], fill=(56, 189, 248, 255))

# 렌즈 내부 "L" 모노그램
lx1, ly1 = cx - 36, cy - 48
lx2, ly2 = cx - 36, cy + 36
lx3, ly3 = cx + 36, cy + 36
p_draw.line([(lx1, ly1), (lx2, ly2)], fill=(248, 250, 252, 255), width=16)
p_draw.line([(lx2 - 8, ly2), (lx3, ly3)], fill=(248, 250, 252, 255), width=16)
p_draw.ellipse([lx1 - 8, ly1 - 8, lx1 + 8, ly1 + 8], fill=(248, 250, 252, 255))
p_draw.ellipse([lx3 - 8, ly3 - 8, lx3 + 8, ly3 + 8], fill=(248, 250, 252, 255))

# 우측 상단 AI 별 (✦ Sparkle)
sx, sy = 348, 136
sr = 32
sparkle_pts = [
    (sx, sy - sr),
    (sx + sr * 0.28, sy - sr * 0.28),
    (sx + sr, sy),
    (sx + sr * 0.28, sy + sr * 0.28),
    (sx, sy + sr),
    (sx - sr * 0.28, sy + sr * 0.28),
    (sx - sr, sy),
    (sx - sr * 0.28, sy - sr * 0.28)
]
p_draw.polygon(sparkle_pts, fill=(56, 189, 248, 255))
p_draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(255, 255, 255, 255))

img.paste(plate, (0, 0), plate)

final_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
final_256.save("assets/ui/app_icon.png")

sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
final_256.save("assets/icon.ico", format="ICO", sizes=sizes)
print("SUCCESS")
