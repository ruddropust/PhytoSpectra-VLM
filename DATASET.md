# Dataset contract

Obtain HyperLeaf2024 from the authors' [Kaggle data page](https://www.kaggle.com/competitions/HyperLeaf2024/data). Follow the original access requirements and terms. The dataset is not included in this source repository.

The pipeline expects an authorized local copy of HyperLeaf2024 with this shape:

```text
<raw-data>/
  train.csv
  images/
    00000.tiff
    ...
```

`train.csv` must contain 1,590 unique `ImageId` values, the continuous targets
`GrainWeight`, `Gsw`, and `PhiPS2`, the categorical `Fertilizer` value, and the
one-hot Cultivar columns `Heerup`, `Kvium`, `Rembrandt`, and `Sheriff`.

Raw images and labels are not redistributed by this repository. Users must
obtain the dataset under its original terms and pass its path with `-RawData`.
