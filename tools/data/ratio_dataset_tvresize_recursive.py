import os

import lmdb

from .ratio_dataset_tvresize import RatioDataSetTVResize


class RatioDataSetTVResizeRecursive(RatioDataSetTVResize):
    """RatioDataSetTVResize variant that recursively discovers LMDB leaves."""

    def _discover_lmdb_dirs(self, root_dir):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(root_dir)

        lmdb_dirs = []
        for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
            if 'data.mdb' in filenames:
                lmdb_dirs.append(dirpath)
        return sorted(lmdb_dirs)

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0

        for root_dir, ratio in zip(data_dir_list, ratio_list):
            lmdb_dirs = self._discover_lmdb_dirs(root_dir)
            if not lmdb_dirs:
                lmdb_dirs = [root_dir]

            valid_count = 0
            for dirpath in lmdb_dirs:
                env = lmdb.open(dirpath,
                                max_readers=32,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
                txn = env.begin(write=False)
                num_samples = txn.get('num-samples'.encode())
                if num_samples is None:
                    env.close()
                    continue

                num_samples = int(num_samples)
                lmdb_sets[dataset_idx] = {
                    'dirpath': dirpath,
                    'env': env,
                    'txn': txn,
                    'num_samples': num_samples,
                    'ratio_num_samples': int(ratio * num_samples),
                }
                dataset_idx += 1
                valid_count += 1

            if valid_count == 0:
                raise RuntimeError(
                    f'No valid LMDB dataset with num-samples found under: {root_dir}'
                )

        return lmdb_sets
