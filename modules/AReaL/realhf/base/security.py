# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").


def read_key(service, name="default"):
    with open(f"/data/marl/keys/{service}/{name}", "r") as f:
        return f.read().strip()
