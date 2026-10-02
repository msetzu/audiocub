import os
from typing import Optional
import json
import pathlib

import pandas
import numpy
import datasets
from tqdm import tqdm

_CACHED = None
_FOLDER_PATH = pathlib.Path(__file__).resolve().parent.parent.parent
_CUB_PATH = _FOLDER_PATH / "CUB_200_2011"


def _build_taxonomy() -> pandas.DataFrame:
    taxonomy = pandas.read_csv(_FOLDER_PATH / "data" / "eBird_taxonomy_v2025.csv")
    taxonomy = taxonomy.rename({c: c.lower() for c in taxonomy.columns}, axis="columns")\
                        .rename({
                            "sci_name": "scientific_name",
                            "primary_com_name": "primary_common_name",
                            "species_code": "species",
                            "taxon_concept_id": "taxon",
                        }, axis="columns")\
                        .drop(["report_as"], axis="columns")\
                        .astype({
                            "species": "string",
                            "taxon": "string",
                            "primary_common_name": "string",
                            "scientific_name": "string",
                            "species_group": "string",
                            "order": "string",
                            "family": "string",
                        })
    # uniform
    taxonomy[taxonomy.select_dtypes(["string"]).columns] = taxonomy.select_dtypes(["string"]).apply(lambda x: x.str.lower().str.replace(r"[-_]", " ", regex=True))

    return taxonomy


def _build_cub(taxonomy: datasets.Dataset, cub_path: Optional[pathlib.Path] = None) -> pandas.DataFrame:
    image2id_df = pandas.read_csv(_FOLDER_PATH / "data" / "images.csv").set_index("file")
    imageid2attributes_df = pandas.read_csv(_FOLDER_PATH / "data" / "image_attribute_labels.csv")
    attribute_names = pandas.read_csv(_FOLDER_PATH / "data" / "attributes.csv")["attribute_name"].values.tolist()

    files = os.listdir(f"{cub_path}/images/")
    cub = list()
    for f in files:
        if not f.endswith(".jpg"):
            continue
        
        image_id = image2id_df.loc[f].values[0]
        class_name = " ".join(f.split("_")[:-2]).lower().replace("_", " ").replace("-", " ")
        attributes_bitvector = imageid2attributes_df.iloc[(image_id - 1) * 312 : (image_id) * 312]["is_present"].astype(bool).tolist()
        with open(f"{cub_path}/images/{f}", "rb") as image:
            image_bytes = image.read()

        cub.append([
            image_id,
            image_bytes,
        ] + attributes_bitvector + [class_name]
        )
    cub = pandas.DataFrame(cub, columns=["image_id", "bytes"] + attribute_names + ["primary_common_name"]).sort_values("image_id")
    cub = cub.astype({"primary_common_name": "string"})
    cub[cub.select_dtypes(["string"]).columns] = cub.select_dtypes(["string"]).apply(lambda x: x.str.lower().str.replace(r"[-_]", " ", regex=True))
    cub = pandas.merge(
        cub,
        taxonomy[["species", "primary_common_name", "category", "taxon", "scientific_name", "order", "family", "species_group"]],
        how="left",
        on="primary_common_name",
    )
    cub["split"] = "train"

    return cub
    

def _build_birdset(taxonomy: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    DATABASES = ["PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE", "XCM", "XCL"][:-1]
    extra_columns = [
        "start_time",
        "end_time",
        "low_freq",
        "high_freq",
        "ebird_code_multilabel",
        "sex",
        "license",
        "event_cluster",
        "peaks",
        "recordist",
        "local_time",
        "order",
        "genus",
        "genus_multilabel",
        "species_group",
        "species_group_multilabel",
        "order_multilabel",
        "ebird_code_secondary",
        "microphone",
        "detected_events",
        "quality",
        "source",
    ]
    birdset_datasets = list()
    for bird_database in tqdm(DATABASES):
        train_df = datasets.load_dataset("DBD-research-group/BirdSet", bird_database, trust_remote_code=True, split="train")\
                    .cast_column("audio", datasets.Audio(sampling_rate=32_000))\
                    .to_pandas()\
                    .drop(extra_columns,axis="columns")
        train_df["split"] = "train"
        if bird_database != "XCM":
            # no test for XCM
            test_df = datasets.load_dataset("DBD-research-group/BirdSet", bird_database, trust_remote_code=True, split="test")\
                    .cast_column("audio", datasets.Audio(sampling_rate=32_000))\
                    .to_pandas()\
                    .drop(extra_columns, axis="columns")
            test_df["split"] = "test"
        
            df = pandas.concat([train_df, test_df], axis="rows")
        else:
            df = train_df
        df["dataset"] = bird_database    
        
        # add all bird classifications
        with open(_FOLDER_PATH / "data" / f"{bird_database}_ebird_codes.json", "r") as log:
            ebird_codes = pandas.DataFrame(json.load(log)["id2label"].items(), columns=["ebird_code", "species"]).astype({"ebird_code": int})

        df = pandas.merge(
            df,
            ebird_codes,
            on="ebird_code",
            how="left",
        )
        df.rename({"name": "species"}, axis="columns", inplace=True)
        df.drop(["ebird_code"], axis="columns", inplace=True)
        
        birdset_datasets.append(df)


    birdset = pandas.concat(birdset_datasets)
    birdset = pandas.merge(
        birdset,
        taxonomy[["species", "primary_common_name", "category", "taxon", "scientific_name", "order", "family", "species_group"]],
        how="left",
        on="species",
    )
    birdset = birdset.astype({"species": "string"})
    birdset[birdset.select_dtypes(["string"]).columns] = birdset.select_dtypes(["string"]).apply(lambda x: x.str.lower().str.replace(r"[-_]", " ", regex=True))
    
    return birdset



def build(cub_path: Optional[pathlib.Path] = _CUB_PATH) -> dict[str, datasets.Dataset]:
    global _CACHED

    if _CACHED is not None:
        return _CACHED

    taxonomy = _build_taxonomy()
    cub, birdset = _build_cub(taxonomy, cub_path), _build_birdset(taxonomy)

    configs = ["species", "primary_common_name", "taxon", "order", "family", "species_group"]
    audiocub_sizes_by_config = list()
    for key in configs:
        cub_frequencies = pandas.DataFrame(cub[key].value_counts())
        birdset_frequencies = pandas.DataFrame(birdset[key].value_counts())

        audiocub_frequencies = pandas.merge(
            cub_frequencies,
            birdset_frequencies,
            how="inner",
            left_index=True,
            right_index=True,
            suffixes=("_cub", "_birdset"),
        )
        audiocub_sizes_by_config.append([key, audiocub_frequencies.shape[0], cub_frequencies.shape[0], birdset_frequencies.shape[0], audiocub_frequencies.shape[0] / cub_frequencies.shape[0], audiocub_frequencies.index.tolist()])

    audiocub_frequencies = pandas.DataFrame(audiocub_sizes_by_config, columns=["key", "audiocub_classes", "cub_classes", "birdset_classes", "ratio", "index"])


    ##################################
    ### Dataset construction #########
    ##################################
    # taxonomy
    taxonomy_ds = datasets.Dataset.from_pandas(taxonomy)

    # cub
    cub_ds_index = numpy.vstack([
        numpy.vstack([cub[row["key"]].isin(row["index"]) for _, row in audiocub_frequencies.iterrows()]),
        (cub["split"] == "train").values
    ]).any(axis=0)
    cub_train_ds = datasets.Dataset.from_pandas(cub[cub_ds_index].drop("split", axis="columns"), preserve_index=False)

    cub_ds_index = numpy.vstack([
        numpy.vstack([cub[row["key"]].isin(row["index"]) for _, row in audiocub_frequencies.iterrows()]),
        (cub["split"] == "test").values
    ]).any(axis=0)
    cub_test_ds = datasets.Dataset.from_pandas(cub[cub_ds_index].drop("split", axis="columns"), preserve_index=False)

    # birdset
    birdset_ds_index = numpy.vstack([
        numpy.vstack([birdset[row["key"]].isin(row["index"]) for _, row in audiocub_frequencies.iterrows()]),
        (birdset["split"] == "train").values
    ]).any(axis=0)
    birdset_train_ds = datasets.Dataset.from_pandas(birdset[birdset_ds_index].drop("split", axis="columns"), preserve_index=False).cast_column("audio", datasets.Audio(sampling_rate=32_000))

    birdset_ds_index = numpy.vstack([
        numpy.vstack([birdset[row["key"]].isin(row["index"]) for _, row in audiocub_frequencies.iterrows()]),
        (birdset["split"] == "test").values
    ]).any(axis=0)
    birdset_test_ds = datasets.Dataset.from_pandas(birdset[birdset_ds_index].drop("split", axis="columns"), preserve_index=False).cast_column("audio", datasets.Audio(sampling_rate=32_000))

    return {
        "taxonomy": taxonomy_ds,
        "cub": {
            "train": cub_train_ds,
            "test": cub_test_ds,
        },
        "birdset": {
            "train": birdset_train_ds,
            "test": birdset_test_ds,
        },
    }
