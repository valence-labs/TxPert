import random

import torch
import numpy as np

from loguru import logger
from gspp.constants import CACHE_DIR


def set_seed(seed) -> None:
    """
    Seeding all the random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_files_from_zenodo(record_id: int, ACCESS_TOKENS: str | None = None)  -> None:
    import requests
    import zipfile
    from tqdm import tqdm

    """
    Download all files from a Zenodo record and save them to the cache directory 
    defined in constants.
    
    Args:
        record_id: The Zenodo record ID.
        ACCESS_TOKENS: Optional access token for authentication.
    """
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Construct the Zenodo API URL for the record
    api_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKENS}"} if ACCESS_TOKENS else {}
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch record metadata: {response.status_code}")
    record_data = response.json()

    # Iterate over each file in the record
    for file_info in record_data.get("files", []):
        file_url = file_info["links"]["self"]
        filename = cache_dir / file_info["key"]
        if not filename.exists():
            logger.info(f"Downloading {file_url} to {filename}")
            with requests.get(file_url, stream=True, headers=headers) as file_response:
                if file_response.status_code == 200:
                    total_size = int(file_response.headers.get("content-length", 0))
                    with (
                        open(filename, "wb") as f,
                        tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=str(filename),
                        ) as progress_bar,
                    ):
                        for chunk in file_response.iter_content(chunk_size=1024):
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                else:
                    raise Exception(
                        f"Failed to download {file_url}, status code: {file_response.status_code}"
                    )

        else:
            logger.info(f"File {filename} already exists. Skipping download.")

    # Unzip any downloaded zip files
    for file_info in record_data.get("files", []):
        filename = cache_dir / file_info["key"]
        if filename.suffix == ".zip":
            with zipfile.ZipFile(filename, "r") as zip_ref:
                extracted_files = [cache_dir / member for member in zip_ref.namelist()]
                if all(file.exists() for file in extracted_files):
                    logger.info(
                        f"All files from {filename} already exist. Skipping unzip."
                    )
                else:
                    logger.info(f"Unzipping {filename}")
                    zip_ref.extractall(cache_dir)
