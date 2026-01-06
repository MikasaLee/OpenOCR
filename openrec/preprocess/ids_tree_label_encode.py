import numpy as np

from tools.utils.ids_to_tree import (
    char_sequence_to_forest,
    ids_to_tree_supervision,
    load_char_to_ids,
)
from .ctc_label_encode import BaseRecLabelEncode


class IDSTreeLabelEncode(BaseRecLabelEncode):
    """Label encoder that also produces IDS-based tree supervision.

    Outputs:
        - label: padded token indices for text (CTC-style, no <sos>/<eos>), shape [max_text_length]
        - length: text length (after filtering OOV and truncation), scalar
        - tree_tokens: list per character, each a list of IDS tokens
        - tree_parents: list per character, each a list of parent indices (0-based, root=-1)
        - tree_relations: list per character, each a list of relation labels
    """

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        char_to_ids_path="tools/utils/dict/visual_c3_ids/char_to_ids.txt",
        add_sos=True,
        add_eos=False,
        label_mode="ids",  # 'ids' | 'char'
        **kwargs,
    ):
        super().__init__(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            lower=kwargs.get("lower", False),
        )
        self.use_space_char = use_space_char
        self.unk_idx = self.dict.get("<unk>")
        self.char2ids = load_char_to_ids(char_to_ids_path)
        # Build inverse mapping IDS string -> char (first occurrence wins)
        self.ids2char = {}
        for ch, ids in self.char2ids.items():
            self.ids2char.setdefault(ids, ch)
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.label_mode = label_mode

    def __call__(self, data):
        text = data["label"]
        if self.lower:
            text = text.lower()

        # Decide parsing mode
        segments = None
        if self.label_mode == "ids":
            segments = [seg for seg in text.split(" ") if seg] if " " in text else [text]
        elif self.label_mode == "char":
            segments = None

        # Case 1: IDS 输入（强制或自动识别）
        if segments is not None:
            if len(segments) == 0:
                return None

            # Build tree supervision per segment
            forest = [
                ids_to_tree_supervision(
                    seg, add_sos=self.add_sos, add_eos=self.add_eos
                )
                for seg in segments
            ]

            # 扁平化 IDS token 序列（包含运算符和部件），可选用空格分隔段
            flat_tokens = []
            for i, seg in enumerate(segments):
                flat_tokens.extend(list(seg))
                if i != len(segments) - 1 and self.use_space_char:
                    flat_tokens.append(" ")

            # 映射到 vocab，OOV -> <unk>（若存在）
            label_ids = []
            for tok in flat_tokens:
                if tok in self.dict:
                    label_ids.append(self.dict[tok])
                elif self.unk_idx is not None:
                    label_ids.append(self.unk_idx)
            if len(label_ids) == 0:
                return None

            g_tokens, g_parents, g_rels = self._build_global_tree(
                forest, insert_space=self.use_space_char
            )

            # 对齐长度截断（CTC标签与树保持一致）
            if len(label_ids) > self.max_text_len:
                label_ids = label_ids[: self.max_text_len]
                g_tokens = g_tokens[: self.max_text_len]
                g_parents = g_parents[: self.max_text_len]
                g_rels = g_rels[: self.max_text_len]
            # 若树比标签短（极端 OOV 情况），补 -1/PAD 保持长度一致
            if len(g_parents) < len(label_ids):
                pad_len = len(label_ids) - len(g_parents)
                g_tokens.extend(["<pad>"] * pad_len)
                g_parents.extend([-1] * pad_len)
                g_rels.extend(["PAD"] * pad_len)
            # 再补到 max_text_len，方便直接 batch
            if len(g_parents) < self.max_text_len:
                pad_len = self.max_text_len - len(g_parents)
                g_tokens.extend(["<pad>"] * pad_len)
                g_parents.extend([-1] * pad_len)
                g_rels.extend(["PAD"] * pad_len)

            data["length"] = np.array(len(label_ids))
            data["label"] = np.array(
                label_ids + [0] * (self.max_text_len - len(label_ids)),
                dtype=np.int64,
            )
            data["tree_tokens"], data["tree_parents"], data["tree_relations"] = (
                g_tokens,
                g_parents,
                g_rels,
            )
            return data

        # Case 2: raw character string
        kept_chars = []
        for ch in text:
            if ch not in self.dict:
                continue
            kept_chars.append(ch)
        if len(kept_chars) == 0:
            return None
        if len(kept_chars) > self.max_text_len:
            kept_chars = kept_chars[: self.max_text_len]

        # token ids for recognition
        label_ids = [self.dict[ch] for ch in kept_chars]
        data["length"] = np.array(len(label_ids))
        data["label"] = np.array(
            label_ids + [0] * (self.max_text_len - len(label_ids)), dtype=np.int64
        )

        # tree supervision per character
        forest = char_sequence_to_forest(
            "".join(kept_chars),
            self.char2ids,
            add_sos=self.add_sos,
            add_eos=self.add_eos,
        )
        g_tokens, g_parents, g_rels = self._build_global_tree(forest)
        # 截断并补齐到 max_text_len，与 label 一样可直接 batch
        if len(g_tokens) > self.max_text_len:
            g_tokens = g_tokens[: self.max_text_len]
            g_parents = g_parents[: self.max_text_len]
            g_rels = g_rels[: self.max_text_len]
        if len(g_tokens) < self.max_text_len:
            pad_len = self.max_text_len - len(g_tokens)
            g_tokens.extend(["<pad>"] * pad_len)
            g_parents.extend([-1] * pad_len)
            g_rels.extend(["PAD"] * pad_len)

        data["tree_tokens"], data["tree_parents"], data["tree_relations"] = (
            g_tokens,
            g_parents,
            g_rels,
        )

        return data

    def add_special_char(self, dict_character):
        # Align with CTC-style blank at index 0 so padding with 0 is safe.
        # Reserve blank at 0; add <unk> to keep alignment when OOV tokens appear.
        return ["blank", "<unk>"] + dict_character

    @staticmethod
    def _strip_root(tokens, parents, relations):
        """Remove leading <sos> and rebase parent indices for alignment with flat IDS tokens."""
        if tokens and tokens[0] == "<sos>":
            tokens = tokens[1:]
            relations = relations[1:]
            new_parents = []
            for p in parents[1:]:
                if p in (-1, 0):
                    new_parents.append(-1)
                else:
                    new_parents.append(p - 1)
            parents = new_parents
        return tokens, parents, relations

    def _build_global_tree(self, forest, insert_space=False):
        """Flatten per-segment trees into a single global index space."""
        g_tokens = []
        g_parents = []
        g_rels = []
        for i, sup in enumerate(forest):
            tokens, parents, rels = self._strip_root(sup.tokens, sup.parents, sup.relations)
            offset = len(g_tokens)
            for t, p, r in zip(tokens, parents, rels):
                g_tokens.append(t)
                g_parents.append(-1 if p == -1 else p + offset)
                g_rels.append(r)
            if insert_space and i != len(forest) - 1:
                g_tokens.append(" ")
                g_parents.append(-1)
                g_rels.append("PAD")
        return g_tokens, g_parents, g_rels
