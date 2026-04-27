from pathlib import Path

import numpy as np

from tools.utils.ids_syntax import DEFAULT_IDC_ARITY


TAN_RELATIONS = [
    'Root',
    'Left',
    'Right',
    'Middle',
    'Above',
    'Below',
    'Inside',
    'Outside',
]

IDC_REL_MAP = {
    '⿰': ['Left', 'Right'],
    '⿱': ['Above', 'Below'],
    '⿲': ['Left', 'Middle', 'Right'],
    '⿳': ['Above', 'Middle', 'Below'],
    '⿴': ['Outside', 'Inside'],
    '⿵': ['Outside', 'Inside'],
    '⿶': ['Outside', 'Inside'],
    '⿷': ['Outside', 'Inside'],
    '⿸': ['Outside', 'Inside'],
    '⿹': ['Outside', 'Inside'],
    '⿺': ['Outside', 'Inside'],
    '⿻': ['Outside', 'Inside'],
}


def _load_token_dict(path):
    token_to_id = {}
    id_to_token = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        token, idx = line.rsplit(' ', 1)
        idx = int(idx)
        token_to_id[token] = idx
        id_to_token[idx] = token
    return token_to_id, id_to_token


def _tokenize_ids(text):
    text = str(text).strip()
    if not text:
        return []
    if ' ' in text:
        return [tok for tok in text.split(' ') if tok]
    return list(text)


class TANLabelEncode(object):

    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 relation_dict_path=None,
                 use_space_char=False,
                 lower=False,
                 **kwargs):
        self.max_text_len = int(max_text_length)
        self.lower = bool(lower)
        self.use_space_char = bool(use_space_char)
        self.dict, self.id_to_token = _load_token_dict(character_dict_path)

        if relation_dict_path is None:
            self.rel_dict = {name: idx for idx, name in enumerate(TAN_RELATIONS)}
        else:
            self.rel_dict, _ = _load_token_dict(relation_dict_path)

        self.none_id = self.dict.get('None', 0)

    def _parse_prefix(self, tokens, pos, parent_idx, parent_token_id,
                      rel_name, nodes):
        if pos >= len(tokens):
            raise ValueError(f'Invalid IDS prefix sequence: {tokens}')

        token = tokens[pos]
        cur_idx = len(nodes)
        token_id = self.dict.get(token)
        if token_id is None:
            raise ValueError(f'Unknown TAN token: {token}')

        relation_id = self.rel_dict.get(rel_name, self.rel_dict['Root'])
        nodes.append({
            'token_id': token_id,
            'parent_idx': parent_idx if parent_idx >= 0 else 0,
            'parent_token_id': parent_token_id if parent_idx >= 0 else self.none_id,
            'relation_id': relation_id,
        })

        arity = DEFAULT_IDC_ARITY.get(token, 0)
        rel_names = IDC_REL_MAP.get(token, [])
        if arity != len(rel_names):
            rel_names = ['Inside'] * arity

        next_pos = pos + 1
        for child_rel in rel_names:
            next_pos = self._parse_prefix(tokens, next_pos, cur_idx, token_id,
                                          child_rel, nodes)
        return next_pos

    def __call__(self, data):
        text = data['label']
        if self.lower:
            text = text.lower()

        tokens = _tokenize_ids(text)
        if len(tokens) == 0:
            return None

        if len(tokens) > self.max_text_len:
            tokens = tokens[:self.max_text_len]

        nodes = []
        try:
            end_pos = self._parse_prefix(tokens, 0, -1, self.none_id, 'Root',
                                         nodes)
        except Exception:
            return None
        if end_pos != len(tokens):
            return None

        node_len = len(nodes)
        if node_len == 0 or node_len > self.max_text_len:
            return None

        label = np.zeros((self.max_text_len, ), dtype=np.int64)
        ly = np.zeros((self.max_text_len, ), dtype=np.int64)
        ly_mask = np.zeros((self.max_text_len, ), dtype=np.float32)
        ry = np.zeros((self.max_text_len, ), dtype=np.int64)
        ry_mask = np.zeros((self.max_text_len, ), dtype=np.float32)
        lp = np.zeros((self.max_text_len, ), dtype=np.int64)
        rp = np.zeros((self.max_text_len, ), dtype=np.int64)
        re = np.zeros((self.max_text_len, ), dtype=np.int64)
        rre = np.zeros((self.max_text_len, ), dtype=np.int64)
        rre_mask = np.zeros((self.max_text_len, ), dtype=np.float32)

        for idx, node in enumerate(nodes):
            label[idx] = node['token_id']
            ly[idx] = node['token_id']
            ly_mask[idx] = 1.0
            ry[idx] = node['parent_token_id']
            ry_mask[idx] = 1.0
            lp[idx] = idx
            rp[idx] = node['parent_idx']
            re[idx] = node['relation_id']
            rre[idx] = node['relation_id']
            rre_mask[idx] = 1.0

        ry_mask[0] = 0.0
        rre_mask[0] = 0.0
        rre_mask[node_len - 1] = 0.0

        data['length'] = np.array(node_len, dtype=np.int64)
        data['label'] = label
        data['ly'] = ly
        data['ly_mask'] = ly_mask
        data['ry'] = ry
        data['ry_mask'] = ry_mask
        data['lp'] = lp
        data['rp'] = rp
        data['re'] = re
        data['rre'] = rre
        data['rre_mask'] = rre_mask
        return data
