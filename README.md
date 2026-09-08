# Quickstart

## Huggingface
The dataset is (soon to be) available on Huggingface at `mstz/audiocub`. To facilitate different tasks, the Birdset and CUB component are split in different datasets:

- `taxonomy`: Complete Birdset taxonomy
- `cub`: The CUB dataset, complete of concepts. Subset of [the original CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/), including only birds indexed by the taxonomy
- `birdset`: The Birdset dataset. Subset of [the original Birdset](https://huggingface.co/datasets/DBD-research-group/BirdSet), including only birds indexed by the taxonomy


```python
from datasets import load_dataset


birdset = load_dataset("mstz/audiocub", "birdset", split="train").cast_column("audio", datasets.Audio(sampling_rate=32_000))
cub = load_dataset("mstz/audiocub", "cub", split="train")
taxonomy = load_dataset("mstz/audiocub", "taxonomy", split="train")
```

## Build
Download CUB

```sh
cd audiocub

# cub
wget "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar xf CUB_200_2011.tgz
rm CUB_200_2011.tgz
mv CUB_200_2011/images/*/*.jpg CUB_200_2011/images/
```
Run the build script
```sh
pip install -r requirements.txt
```
then the dataset is available as a `dict[str, datasets.Dataset]` through
```python
import audiocub

data = audiocub.build()
# {
#     "taxonomy": Dataset(...),
#     "cub": {
#         "train": Dataset(...),
#         "test": Dataset(...),
#     },
#     "birdset": {
#         "train": Dataset(...),
#         "test": Dataset(...),
#     },
# }
# ```