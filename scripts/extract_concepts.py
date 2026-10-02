import os
import pathlib
import sys
from datetime import datetime
from math import ceil

from joblib import Parallel, delayed
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy
import pandas
from huggingface_hub import whoami
# from pandarallel import pandarallel

sys.path.append("../src")

from audiocub.concepts.audio import extract_all_audio_concepts, extract_all_audio_concepts_extended
from audiocub.concepts.audio import SR
from audiocub.dataset import build


_OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "data" / "output"
_CHUNK_SIZE = 10_000
_MAX_DURATION = 300  # in seconds, i.e., 5min


# pandarallel.initialize(progress_bar=True, use_memory_fs=False)


def extract():
    user = whoami(token=os.environ["HUGGINGFACE_TOKEN"])

    print("Loading Audiocub...")
    data = build()
    print("Loading Birdset...")
    birdset = pandas.concat([
        data["birdset"]["train"].to_pandas(),
        data["birdset"]["test"].to_pandas(),
    ])
    print("Loading audio...")
    birdset["audio"] = None
    print(f"Filtering from {birdset.shape[0]}...")
    filtering_features = [
        "category",
        "family",
        "order",
        "primary_common_name",
        "scientific_name",
        "species",
        "species_group",
        "taxon"
    ]
    masks = numpy.vstack([
        birdset[feature].isin(data["cub"]["train"].to_pandas()[feature].unique()) for feature in filtering_features
    ])
    # loosest mask any filtering feature is active
    mask = masks.any(axis=0)
    mask = masks[3]  # primary common name
    index = numpy.argwhere(mask).flatten()
    birdset = birdset.iloc[index]
    print(f"Filtered down to {birdset.shape[0]}...")

    # ram saving :(
    complete_audio_concepts = list()
    complete_audio_concepts_extended = list()
    chunks_nr = ceil(birdset.shape[0] / _CHUNK_SIZE)
    for chunk_nr in range(chunks_nr):
        if (_OUTPUT_PATH / f"audio_concepts_chunk_{chunk_nr}.csv").exists()\
            and (_OUTPUT_PATH / f"audio_concepts_extended_chunk_{chunk_nr}.csv").exists():
            continue

        print(f"Chunk {chunk_nr} / {chunks_nr}")
        chunk = birdset.iloc[chunk_nr * _CHUNK_SIZE : (chunk_nr + 1) * _CHUNK_SIZE]

        files = chunk["filepath"].tolist()
        results = Parallel(n_jobs=-1, batch_size=30, return_as="generator")(
            delayed(extract_all_audio_concepts_extended)(path) for path in files
        )
        results = list(tqdm(results, total=len(files)))
        audio_concepts_extended = pandas.DataFrame(results)
        audio_concepts_extended.to_csv(_OUTPUT_PATH / f"audio_concepts_extended_chunk_{chunk_nr}.csv", index=False)

        # clean memory
        chunk["audio"] = None
        birdset["audio"] = None

        # complete_audio_concepts.append(audio_concepts)
        complete_audio_concepts_extended.append(audio_concepts_extended)

    pandas.concat(complete_audio_concepts, axis="rows").to_csv(_OUTPUT_PATH / "audio_concepts_extended.csv", index=False)


if __name__ == "__main__":
    extract()

