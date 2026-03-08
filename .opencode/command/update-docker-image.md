---
description: Update the `Dockerfile` inside this repo given the package version from the input, and use github CI to build and test the new image.
---

## Usage

```
/update-docker-image $VERSION
```

**Arguments (`$VERSION`):** a list of pinned package versions, such as "sglang==0.5.9
vllm==0.10.1 torch==2.10.1"

## Architecture

The Dockerfile produces two image variants from a single file:

- `ghcr.io/inclusionai/areal-runtime:{tag}-sglang` — SGLang inference backend
- `ghcr.io/inclusionai/areal-runtime:{tag}-vllm` — vLLM inference backend

Both variants share the same base image (`lmsysorg/sglang:…`) and identical layers up to
STAGE 3. Only `ARG VARIANT` (declared late for cache efficiency) controls which
inference backend is installed via `--extra ${VARIANT}`.

The `latest` tag always points to the sglang variant.

## Workflow

1. **Validate versions.** Update the version requirements in @pyproject.toml according
   to the input. Validate that the provided versions exist in the pip registry.
   Otherwise, exit and raise an error report to the user. Keep other dependency versions
   unchanged in this step.

1. **Check upstream dependency compatibility.** For the following packages, browse the
   GitHub repo and check for dependency version mismatches with AReaL:

   - For sglang, check
     `https://github.com/sgl-project/sglang/blob/v${version}/python/pyproject.toml`
   - For vllm, check
     `https://github.com/vllm-project/vllm/blob/v${version}/pyproject.toml`

1. **Resolve dependency conflicts** and report to user.

   If there's no inconsistency between the above packages, and it only conflicts with
   AReaL, update AReaL's version.

   If the above packages have mutual conflict, summarize and report to user, then you
   MUST ask the user for resolution.

   Output format:

   ```
   Summary

   ---
   Updated Packages (no actions required):
   - ${name}, ${packageA} requires ${packageAVersion}, ${packageB} requires ${packageBVersion}, AReaL specified ${version}, updated to ${version}
   - ...

   ---
   Mismatched Packages (need to resolve):

   - ${name}, ${packageA} requires ${packageAVersion}, ${packageB} requires ${packageBVersion}
   - ...
   ```

1. **Update @pyproject.toml** according to the user's conflict resolution. You may use
   `override-dependencies` in `[tool.uv]` to force-pin versions where needed. Remember
   that `sglang` and `vllm` are declared as **conflicting extras** — never install both.

1. **Validate** that the conflicts in step 3 have been all resolved. If not, return to
   step 3 and you MUST ask the user again.

1. **Lock dependencies.** Run `uv lock` to regenerate `uv.lock`. If errors occur, return
   to step 3 — you must ask the user for resolution before modifying and trying again.

1. **Update the Dockerfile** if needed. The Dockerfile uses only `ARG VARIANT` (no
   `ARG BASE_IMAGE`) to select between sglang and vllm. All layers before the VARIANT
   declaration are shared between both variants for Docker cache efficiency.

   Only modify the Dockerfile if the base image, system packages, or build steps need to
   change (e.g., new base image URL, new CUDA version). Do NOT modify it just for
   pyproject.toml version bumps — `uv pip install -r pyproject.toml` handles that
   automatically.

1. **Create a PR and trigger CI.** Use the `/create-pr` command to create a PR, then
   trigger the CI workflow manually via `.github/workflows/build-docker-image.yml`.

   The docker build CI builds both sglang and vllm images, then automatically triggers
   testing on each. Debug until the overall workflow succeeds.

   If you encounter issues that cannot be resolved, ask the user for help.
