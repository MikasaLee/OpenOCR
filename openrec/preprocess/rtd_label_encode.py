"""
RTDLabelEncode: AR-style label encoding with random token replacement
for Replaced Token Detection (RTD) training.

For each sample:
  1. Encode text to index sequence (same as ARLabelEncode).
  2. Randomly corrupt ~replace_ratio of positions.
  3. Generate binary RTD labels (1=replaced, 0=original).

Output keys added to data dict:
  - 'label'     : [BOS, y~_1, ..., y~_T, EOS, PAD, ...]  (int64 array)
  - 'length'    : T  (actual text length, scalar)
  - 'rtd_label' : [e_1, ..., e_T, 0, 0, ...]             (int64 array, padded)

Replacement strategy (end-to-end):
  - Pass `char_to_ids_path` (char<TAB>ids_seq) to automatically build
    a similar-character mapping via IDS Levenshtein edit distance at init
    time.  Uses `rapidfuzz.process.cdist` for fast batch computation.
  - `hard_ratio` controls the proportion of IDS-hard negatives among
    all replacements (default 0.7 = 70% hard, 30% random).
  - Falls back to uniform random replacement for chars without IDS
    or without similar neighbors.
"""

import random

import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


class RTDLabelEncode(BaseRecLabelEncode):

    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        replace_ratio=0.15,
        hard_ratio=0.7,
        char_to_ids_path=None,
        sim_top_k=10,
        sim_max_dist=3,
        sim_max_norm_dist=0.5,
        sim_min_ids_len=2,
        **kwargs,
    ):
        super().__init__(max_text_length, character_dict_path, use_space_char)
        self.replace_ratio = replace_ratio
        self.hard_ratio = hard_ratio

        # Build vocabulary list for random replacement (exclude special tokens)
        # character order: [EOS] + real_chars + [BOS, PAD]
        # real chars are indices 1 .. len(character)-3
        self.real_char_indices = list(range(1, len(self.character) - 2))

        # Build similar-character mapping from char-to-IDS mapping
        self.similar_map = {}
        if char_to_ids_path:
            self._build_similar_map(
                char_to_ids_path,
                top_k=sim_top_k,
                max_dist=sim_max_dist,
                max_norm_dist=sim_max_norm_dist,
                min_ids_len=sim_min_ids_len,
            )

    # ------------------------------------------------------------------
    # Similar-character map construction (runs once at init)
    # ------------------------------------------------------------------

    def _build_similar_map(self, char_to_ids_path, top_k=10,
                           max_dist=3, max_norm_dist=0.5, min_ids_len=2):
        """Build similar-char mapping from char-to-IDS file via edit distance.

        Uses `rapidfuzz.process.cdist` for batch C-level computation so the
        full pairwise matrix is built in seconds even for ~3000 characters.
        """
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import cdist as rf_cdist

        # 1. Load char -> IDS mapping
        char_to_ids = {}
        with open(char_to_ids_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').strip('\r\n')
                if not line or '\t' not in line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    char_to_ids[parts[0]] = parts[1]

        # 2. Filter to vocab chars with valid IDS decomposition
        indices = []   # vocab index for each row
        ids_list = []  # corresponding IDS string
        for idx in self.real_char_indices:
            ch = self.character[idx]
            if ch in char_to_ids:
                ids_str = char_to_ids[ch]
                if len(ids_str) >= min_ids_len:
                    indices.append(idx)
                    ids_list.append(ids_str)

        n = len(indices)
        if n == 0:
            return

        # 3. Batch pairwise Levenshtein via cdist (C-level, very fast)
        dist_matrix = rf_cdist(
            ids_list, ids_list,
            scorer=Levenshtein.distance,
            score_cutoff=max_dist,
            workers=-1,
            dtype=np.int32,
        )
        # Entries > max_dist are set to max_dist+1 by cdist.
        # Exclude self-matches.
        np.fill_diagonal(dist_matrix, max_dist + 1)

        # Pre-compute lengths for normalised distance filtering
        lens = np.array([len(s) for s in ids_list], dtype=np.float64)

        # 4. Extract top-K neighbors per character
        for i in range(n):
            row = dist_matrix[i]
            mask = row <= max_dist
            if not mask.any():
                continue
            js = np.where(mask)[0]
            dists = row[js].astype(np.float64)
            norm_dists = dists / np.maximum(lens[i], lens[js])
            valid = norm_dists <= max_norm_dist
            if not valid.any():
                continue
            js = js[valid]
            dists = dists[valid]
            norm_dists = norm_dists[valid]
            order = np.lexsort((norm_dists, dists))[:top_k]
            self.similar_map[indices[i]] = [int(indices[j]) for j in js[order]]

    # ------------------------------------------------------------------
    # Replacement logic
    # ------------------------------------------------------------------

    def _get_replacement(self, orig_idx):
        """Get a replacement index for the given original character index."""
        use_hard = (random.random() < self.hard_ratio
                    and orig_idx in self.similar_map)
        if use_hard:
            candidates = self.similar_map[orig_idx]
            return random.choice(candidates)
        else:
            # Random replacement from vocabulary (exclude the original)
            while True:
                new_idx = random.choice(self.real_char_indices)
                if new_idx != orig_idx:
                    return new_idx

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None

        text_len = len(text)
        data['length'] = np.array(text_len)

        # ------- random replacement -------
        rtd_label = [0] * text_len
        corrupted = list(text)  # copy

        # decide how many positions to replace
        if self.replace_ratio > 0:
            num_replace = max(1, int(text_len * self.replace_ratio))
            if num_replace > text_len:
                num_replace = text_len
        else:
            num_replace = 0

        if num_replace > 0:
            replace_positions = random.sample(range(text_len), num_replace)
            for pos in replace_positions:
                new_idx = self._get_replacement(corrupted[pos])
                corrupted[pos] = new_idx
                rtd_label[pos] = 1

        # ------- format as AR label: [BOS, y~_1, ..., y~_T, EOS, PAD, ...] ------
        label = ([self.dict[self.BOS]] + corrupted + [self.dict[self.EOS]])
        label = label + [self.dict[self.PAD]] * (self.max_text_len + 2 - len(label))
        data['label'] = np.array(label, dtype=np.int64)

        # ------- RTD label: [e_1, ..., e_T, 0, 0, ...] padded -------
        rtd_label = rtd_label + [0] * (self.max_text_len - len(rtd_label))
        data['rtd_label'] = np.array(rtd_label, dtype=np.int64)

        return data

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.PAD]
        return dict_character


if __name__ == '__main__':
    import sys, os, time

    DICT_PATH = './tools/utils/dict/visual_c3_ids/char_dict.txt'
    IDS_PATH  = './tools/utils/dict/visual_c3_ids/char_to_ids.txt'

    if not os.path.exists(DICT_PATH):
        print(f'[SKIP] {DICT_PATH} not found, run from project root.')
        sys.exit(0)

    print('=' * 60)
    print('TEST 1: Basic init WITHOUT similar map')
    print('=' * 60)
    enc_basic = RTDLabelEncode(
        max_text_length=15,
        character_dict_path=DICT_PATH,
        use_space_char=True,
        replace_ratio=0.15,
    )
    assert len(enc_basic.similar_map) == 0, 'similar_map should be empty'
    BOS_IDX = enc_basic.dict[enc_basic.BOS]
    EOS_IDX = enc_basic.dict[enc_basic.EOS]
    PAD_IDX = enc_basic.dict[enc_basic.PAD]
    vocab_size = len(enc_basic.character)
    print(f'  vocab_size={vocab_size}, BOS={BOS_IDX}, EOS={EOS_IDX}, PAD={PAD_IDX}')
    print('  [PASS] Basic init OK\n')

    print('=' * 60)
    print('TEST 2: Init WITH similar map (IDS cdist)')
    print('=' * 60)
    t0 = time.time()
    enc = RTDLabelEncode(
        max_text_length=15,
        character_dict_path=DICT_PATH,
        use_space_char=True,
        replace_ratio=0.15,
        hard_ratio=0.7,
        char_to_ids_path=IDS_PATH,
    )
    t1 = time.time()
    print(f'  Build time: {t1-t0:.2f}s, entries: {len(enc.similar_map)}')
    assert len(enc.similar_map) > 0, 'similar_map should NOT be empty'
    for idx, sims in list(enc.similar_map.items())[:3]:
        ch = enc.character[idx]
        sim_chars = [enc.character[s] for s in sims[:5]]
        print(f'    {ch} -> {sim_chars}')
    print('  [PASS] Similar map build OK\n')

    print('=' * 60)
    print('TEST 3: Encoding + replacement  (detailed before/after)')
    print('=' * 60)
    test_texts = ['你好世界', '今天天气不错', '测试替换效果展示', 'A']
    for text in test_texts:
        # --- encode original (no replacement) for comparison ---
        orig_indices = enc.encode(text)
        if orig_indices is None:
            print(f'  "{text}" -> None (encode failed)'); continue
        orig_chars = [enc.character[i] for i in orig_indices]

        # --- encode with replacement ---
        data = {'label': text}
        out = enc(data)
        if out is None:
            print(f'  "{text}" -> None'); continue
        label, length, rtd_label = out['label'], out['length'], out['rtd_label']

        # shape / boundary assertions
        assert label.shape == (17,), f'label shape: {label.shape}'
        assert rtd_label.shape == (15,), f'rtd shape: {rtd_label.shape}'
        assert label[0] == BOS_IDX
        text_end = 1 + int(length)
        assert label[text_end] == EOS_IDX
        for p in range(text_end + 1, len(label)):
            assert label[p] == PAD_IDX
        for p in range(int(length), len(rtd_label)):
            assert rtd_label[p] == 0

        # --- extract corrupted chars from label (skip BOS at pos 0) ---
        corrupted_indices = [int(label[1 + i]) for i in range(int(length))]
        corrupted_chars = [enc.character[i] for i in corrupted_indices]

        # --- print detailed comparison ---
        n_replaced = int(rtd_label[:int(length)].sum())
        print(f'\n  原文: "{text}"  (len={int(length)}, 替换数={n_replaced})')
        print(f'  原始字符:   {orig_chars}')
        print(f'  替换后字符: {corrupted_chars}')
        print(f'  RTD标签:    {rtd_label[:int(length)].tolist()}')

        # per-position detail
        for i in range(int(length)):
            flag = '✗ 被替换' if rtd_label[i] == 1 else '✓ 保持'
            extra = ''
            if rtd_label[i] == 1:
                # check if it was a hard (IDS-similar) or random replacement
                if orig_indices[i] in enc.similar_map and corrupted_indices[i] in enc.similar_map.get(orig_indices[i], []):
                    extra = ' [hard/形近]'
                else:
                    extra = ' [random/随机]'
            print(f'    pos {i}: "{orig_chars[i]}" → "{corrupted_chars[i]}"  {flag}{extra}')

    print('\n  [PASS] Encoding OK\n')

    print('=' * 60)
    print('TEST 4: replace_ratio=0')
    print('=' * 60)
    enc0 = RTDLabelEncode(max_text_length=15, character_dict_path=DICT_PATH,
                          use_space_char=True, replace_ratio=0.0)
    out = enc0({'label': '\u6d4b\u8bd5'})
    assert out is not None and int(out['rtd_label'].sum()) == 0
    print('  [PASS] Zero-replacement OK\n')

    print('=' * 60)
    print('ALL RTDLabelEncode TESTS PASSED')
    print('=' * 60)
