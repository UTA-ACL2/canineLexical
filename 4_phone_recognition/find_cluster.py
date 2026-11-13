# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import time

from sklearn.metrics import calinski_harabasz_score, silhouette_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat,
    valid_feat,
    km_path,
    n_clusters,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    # feat = load_feature(feat_dir, split, nshard, seed, percent)

    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )

    km_model.fit(feat)
    s1 = km_model.inertia_

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")
    s12 = inertia

    y = km_model.predict(valid_feat)
    s2 = silhouette_score(valid_feat, y)
    s3 = calinski_harabasz_score(valid_feat, y)
    return s1, s12, s2, s3


if __name__ == "__main__":
    save_path = './find_cluster_from_iter1_80k_6th.txt'

    valid_feat = load_feature(feat_dir='./feat_80k_6th',
                              split='valid',
                              nshard=1,
                              seed=0,
                              percent=-1)
    dtype = valid_feat.dtype
    valid_feat = valid_feat.tolist()
    valid_count = len(valid_feat)
    random.seed(42)
    random.shuffle(valid_feat)
    valid_feat = np.array(valid_feat, dtype=dtype)[0: int(valid_count / 10)]

    print('valid ok')

    feat = load_feature(feat_dir='./feat_80k_6th',
                        split='train',
                        nshard=1,
                        seed=0,
                        percent=-1)

    print('feat ok')

    for n_clusters in range(160, 300, 10):
        name = f'iter1_{str(n_clusters)}.pkl'
        print(name)
        print(n_clusters)
        s1, s12, s2, s3 = learn_kmeans(feat=feat,
                                  valid_feat=valid_feat,
                                  km_path=None,
                                  n_clusters=n_clusters,
                                  seed=0,
                                  init="k-means++",
                                  max_iter=1000,
                                  batch_size=100000,
                                  tol=0.0,
                                  n_init=20,
                                  reassignment_ratio=0.001,
                                  max_no_improvement=500)
        with open(save_path, 'a+', encoding='utf-8') as file:
            file.write(f'{name},{str(n_clusters)},{str(s1)},{str(s12)},{str(s2)},{str(s3)}\n')



