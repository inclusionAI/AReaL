def download_from_huggingface(
    repo_id: str, filename: str, revision: str = "main", repo_type: str = "dataset"
) -> str:
    """
    Download a file from a HuggingFace Hub repository.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Please install huggingface_hub to use this function: pip install huggingface_hub"
        )

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
    )


def load_hf_or_local_file(path: str) -> str:
    """
    Load a file from a HuggingFace Hub repository or a local file.
    hf://<org>/<repo>/<filename>
    hf://<org>/<repo>@<revision>/<filename>

    e.g,
    hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl
    =>
    repo_type = dataset
    repo_id = inclusionAI/AReaL-RL-Data
    filename = data/boba_106k_0319.jsonl
    revision = main
    =>
    /root/.cache/huggingface/hub/models--inclusionAI--AReaL-RL-Data/data/boba_106k_0319.jsonl
    """
    path = str(path)
    if path.startswith("hf://") or path.startswith("hf-dataset://"):
        repo_type = "dataset" if path.startswith("hf-dataset://") else "model"
        hf_path = path.strip().split("://")[1]
        hf_org, hf_repo, filename = hf_path.split("/", 2)
        repo_id = f"{hf_org}/{hf_repo}"
        revision = "main"
        if "@" in repo_id:
            repo_id, revision = repo_id.split("@", 1)
        return download_from_huggingface(repo_id, filename, revision)
    return path
