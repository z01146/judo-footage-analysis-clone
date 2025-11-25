""""Find files in the retrain folder."""

import hashlib
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def compute_sha1(path):
    return hashlib.sha1(path.open("rb").read()).hexdigest()


def collect_paths(src, dst, cores=8):
    # let's find the files in the retrain root we need to deal with
    collected = []
    with Pool(cores) as p:
        paths = list(src.glob("**/*.png"))
        hashes = set(tqdm(p.imap(compute_sha1, paths), total=len(paths)))

        # now let's find the paths in the interesting root that have the same hash
        paths = list(dst.glob("**/*.png"))
        for hash, path in tqdm(
            zip(p.imap(compute_sha1, paths), paths), total=len(paths)
        ):
            if hash in hashes:
                collected.append(path)
    return collected


def normalize_path(paths, relative_root):
    res = []
    for path in paths:
        relpath = path.relative_to(relative_root)
        # remove the unwanted suffix from the filename
        relpath = relpath.parent / f"{relpath.stem.split('_')[0]}.jpg"
        res.append(relpath.as_posix())
    return res


def main():
    data_root = Path("/cs-share/pradalier/tmp/judo/data")
    retrain_root = data_root / "referee_v2_sorted" / "referee_retrain"
    reference_root = data_root / "referee_v2"
    annotation_path = (
        data_root / "annotations" / "project-2-at-2024-04-12-20-17-8b63f47c.json"
    )

    df = pd.read_json(annotation_path, orient="records")
    df["image"] = (
        df["data"]
        .apply(lambda x: x["image"].replace("http://localhost:8080/data/frames/", ""))
        .astype(str)
    )
    subset = df[["id", "image"]]
    # now print out the list of images
    collected = collect_paths(retrain_root, reference_root)
    normalized = normalize_path(collected, reference_root)
    with (retrain_root / "original_path_2.txt").open("w") as f:
        for path in normalized:
            f.write(f"{path}\n")

    filter_df = subset[subset["image"].isin(normalized)]
    print(filter_df)
    filter_df.to_csv(retrain_root / "annotation_to_retrain_2.csv")


if __name__ == "__main__":
    main()
