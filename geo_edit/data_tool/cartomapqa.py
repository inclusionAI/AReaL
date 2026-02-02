from datasets import Dataset, Features, Value, Sequence, Image as HFImage
import os, glob, json

json_dir = r"..\CartoMapQA\Dataset\CartoMapQA\MapNavigation\_routes"
image_root = r"..\CartoMapQA\Dataset\CartoMapQA\MapNavigation\_maps"
image_ext = ".png"
out_parquet = r"..\CartoMapQA\MapNavigation_dataset.parquet"

examples = []
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

for i, fp in enumerate(json_files):
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    base = os.path.splitext(os.path.basename(fp))[0]
    image_path = os.path.join(image_root, f"{base}{image_ext}")
    with open(image_path, "rb") as imf:
        b = imf.read()

    obj["id"] = i
    obj["image"] = {"bytes": b, "path": None}
    examples.append(obj)

features = Features({
    "origin": Sequence(Value("float64")),
    "destination": Sequence(Value("float64")),
    "facing": Value("string"),
    "travel_mode": Value("string"),
    "route_directions": Value("string"),
    "route_node_id": Sequence(Value("int64")),
    "area": Value("string"),
    "zoom_level": Value("int64"),
    "id": Value("int64"),
    "image": HFImage(),
})

ds = Dataset.from_list(examples, features=features)
ds.to_parquet(out_parquet)
print("Saved parquet:", out_parquet)
