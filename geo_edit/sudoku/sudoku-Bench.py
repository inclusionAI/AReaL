from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PIL import Image
from io import BytesIO
import base64

url = "https://sudokupad.app/cl59hc00yp"


driver = webdriver.Chrome()
driver.set_window_size(1400, 900)
driver.get(url)

wait = WebDriverWait(driver, 20)

# 1) 关闭开始弹框
start_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Start Puzzle')]")))
start_btn.click()
wait.until(EC.invisibility_of_element_located((By.XPATH, "//button[contains(., 'Start Puzzle')]")))

# 2) 等棋盘出现并有尺寸
board = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#board")))
wait.until(lambda d: board.is_displayed() and board.size["width"] > 100 and board.size["height"] > 100)

# 3) 获取 board 在“整页文档坐标系”里的位置（不是视口坐标）
info = driver.execute_script(
    "const el = arguments[0];"
    "const r = el.getBoundingClientRect();"
    "return {"
    "  x: r.left + window.scrollX,"
    "  y: r.top + window.scrollY,"
    "  w: r.width,"
    "  h: r.height,"
    "  dpr: window.devicePixelRatio"
    "};",
    board
)

x = max(info["x"] +16, 0)
y = max(info["y"] + 16, 0)
w = info["w"]-32
h = info["h"]-32
dpr = info["dpr"]

# 4) 用 CDP 截取“整页”截图（包含视口外）
shot = driver.execute_cdp_cmd("Page.captureScreenshot", {
    "format": "png",
    "captureBeyondViewport": True
})
png_bytes = base64.b64decode(shot["data"])

img = Image.open(BytesIO(png_bytes))
img_w, img_h = img.size

# 5) 按 DPR 把 CSS 像素坐标换算成图片像素坐标，并裁剪
left = int(x * dpr)
top = int(y * dpr)
right = int((x + w) * dpr)
bottom = int((y + h) * dpr)

left = max(left, 0)
top = max(top, 0)
right = min(right, img_w)
bottom = min(bottom, img_h)

img.crop((left, top, right, bottom)).save("grid_pad_fullpage.png")

driver.quit()
