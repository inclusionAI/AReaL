from datasets import load_dataset, Image as HFImage
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PIL import Image
from io import BytesIO
import base64
import os
from time import sleep


def capture_board_pil(driver, url: str, inset: int = 16) -> Image.Image:
    driver.get(url)
    wait = WebDriverWait(driver, 20)
    sleep(7)  # 等待页面加载
    # 关闭/进入：点 Start Puzzle（如果存在）
    btns = driver.find_elements(By.XPATH, "//button[contains(., 'Start Puzzle')]")
    if len(btns) > 0:
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Start Puzzle')]"))).click()
        wait.until(EC.invisibility_of_element_located((By.XPATH, "//button[contains(., 'Start Puzzle')]")))

    # 等棋盘出现并有尺寸
    board = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#board")))
    wait.until(lambda d: board.is_displayed() and board.size["width"] > 100 and board.size["height"] > 100)

    # 获取 board 在整页坐标系中的位置，并做 inset 裁剪
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

    x = max(info["x"] -inset, 0)
    y = max(info["y"] -inset, 0)
    w = max(info["w"]+inset*2, 1)
    h = max(info["h"]+inset*2, 1)
    dpr = info["dpr"]

    # 全页截图（包含视口外）
    shot = driver.execute_cdp_cmd("Page.captureScreenshot", {
        "format": "png",
        "captureBeyondViewport": True
    })
    full_png = base64.b64decode(shot["data"])
    img = Image.open(BytesIO(full_png))

    left = int(x * dpr)
    top = int(y * dpr)
    right = int((x + w) * dpr)
    bottom = int((y + h) * dpr)

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, img.size[0])
    bottom = min(bottom, img.size[1])

    return img.crop((left, top, right, bottom))


def pil_to_png_bytes(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main(
    input_parquet: str,
    output_parquet: str,
    image_dir: str,
    url_key: str = "sudokupad_url",
    inset: int = 16,
):
    os.makedirs(image_dir, exist_ok=True)

    ds = load_dataset("parquet", data_files=input_parquet, split="train")

    driver = webdriver.Chrome()
    driver.set_window_size(1400, 900)

    image_paths = []
    images = []
    from tqdm import tqdm
    
    for i, ex in enumerate(tqdm(ds)):
        url = ex[url_key]
        puzzle_id=ex["puzzle_id"]
        filename = f"{puzzle_id}.png"
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            board_im = Image.open(path)
            image_paths.append(path)
            images.append({"bytes": pil_to_png_bytes(board_im)})
            continue
        board_im = capture_board_pil(driver, url, inset=inset)
        
        board_im.save(path, format="PNG")

        image_paths.append(path)
        images.append({"bytes": pil_to_png_bytes(board_im)})

    driver.quit()

    ds2 = ds.add_column("board_image_path", image_paths)
    ds2 = ds2.add_column("board_image", images).cast_column("board_image", HFImage())
    ds2.to_parquet(output_parquet)


if __name__ == "__main__":
    main(
        input_parquet="..\\..\\data\\sudoku_dir\\sudoku_bench\\challenge_100\\test-00000-of-00001.parquet",
        output_parquet="..\\..\\data\\sudoku_dir\\sudoku_bench\\challenge_100_with_image\\test-00000-of-00001.parquet",
        image_dir="sudokupad_boards",
        url_key="sudokupad_url",
        inset=16,
    )

