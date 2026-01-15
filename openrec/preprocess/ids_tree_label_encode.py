import numpy as np

from tools.utils.ids_to_tree import (
    char_sequence_to_forest,
    ids_to_tree_supervision,
    load_char_to_ids,
)
from .ctc_label_encode import BaseRecLabelEncode


class IDSTreeLabelEncode(BaseRecLabelEncode):
    """基于 IDS 的树监督 + seq2seq 文本标签编码器。

    与 TAMER 对齐的特殊符号约定：
    - <pad>=0, <sos>=1, <eos>=2, <unk>=3

    输出：
      - label: [<sos>] + 内容 + [<eos>] + <pad>...，固定长度 2 + max_text_len
      - length: 内容长度（不含 <sos>/<eos>），用于切片 2 + max_len
            - tree_tokens/tree_parents/tree_relations: 与 label 对齐（同长度），保留全局根 <sos>；
                根/空间/PAD/<eos> 的 parent 统一为 -1。
    """

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        char_to_ids_path="tools/utils/dict/visual_c3_ids/char_to_ids.txt",
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
        # 内部总是用全局 <sos>/<eos>，分段树不再各自加根
        self.add_sos = True
        self.add_eos = True
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
            # 分段树不加本地 <sos>/<eos>，将在全局阶段合并到统一根
            forest = [
                ids_to_tree_supervision(
                    seg, add_sos=False, add_eos=False
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

            # seq2seq：内容截断，包上 <sos>/<eos> 并补齐
            content_len = min(len(label_ids), self.max_text_len)
            SOS, EOS, PAD = self.dict["<sos>"], self.dict["<eos>"], self.dict["<pad>"]
            label_seq = [SOS] + label_ids[:content_len] + [EOS]
            total_len = 2 + self.max_text_len
            if len(label_seq) < total_len:
                label_seq += [PAD] * (total_len - len(label_seq))

            # 全局树：保留统一 <sos> 根；各段根保持 -1；对齐到 [<sos>] + 内容[:content_len] + [<eos>]
            g_tokens, g_parents, g_rels = self._build_global_tree(
                forest,
                content_len=content_len,
                insert_space=self.use_space_char,
                with_global_sos=True,
                with_global_eos=True,
            )
            # 补到统一长度 2 + max_text_len
            if len(g_tokens) < total_len:
                pad_len = total_len - len(g_tokens)
                g_tokens.extend(["<pad>"] * pad_len)
                g_parents.extend([-1] * pad_len)
                g_rels.extend(["PAD"] * pad_len)

            data["length"] = np.array(content_len)
            data["label"] = np.array(label_seq, dtype=np.int64)
            data["tree_tokens"], data["tree_parents"], data["tree_relations"] = (
                np.array(g_tokens),
                np.array(g_parents),
                np.array(g_rels),
            )
            # print(f'data:{data}') # debug
            return data

        # Case 2: raw character string
        kept_chars = []
        for ch in text:
            if ch not in self.dict:
                continue
            kept_chars.append(ch)
        if len(kept_chars) == 0:
            return None
        # token ids for recognition (内容部分)
        content_ids = [self.dict[ch] for ch in kept_chars]
        content_len = min(len(content_ids), self.max_text_len)
        SOS, EOS, PAD = self.dict["<sos>"], self.dict["<eos>"], self.dict["<pad>"]
        label_seq = [SOS] + content_ids[:content_len] + [EOS]
        total_len = 2 + self.max_text_len
        if len(label_seq) < total_len:
            label_seq += [PAD] * (total_len - len(label_seq))

        # tree supervision per character（分字符树，不加本地根）
        forest = char_sequence_to_forest(
            "".join(kept_chars),
            self.char2ids,
            add_sos=False,
            add_eos=False,
        )
        g_tokens, g_parents, g_rels = self._build_global_tree(
            forest,
            content_len=content_len,
            insert_space=False,
            with_global_sos=True,
            with_global_eos=True,
        )
        if len(g_tokens) < total_len:
            pad_len = total_len - len(g_tokens)
            g_tokens.extend(["<pad>"] * pad_len)
            g_parents.extend([-1] * pad_len)
            g_rels.extend(["PAD"] * pad_len)

        data["length"] = np.array(content_len)
        data["label"] = np.array(label_seq, dtype=np.int64)
        data["tree_tokens"], data["tree_parents"], data["tree_relations"] = (
            np.array(g_tokens),
            np.array(g_parents),
            np.array(g_rels),
        )

        return data

    def add_special_char(self, dict_character):
        # 与 TAMER 对齐：<pad>=0, <sos>=1, <eos>=2, <unk>=3
        return ["<pad>", "<sos>", "<eos>", "<unk>"] + dict_character

    def _build_global_tree(
        self,
        forest,
        content_len: int,
        insert_space: bool = False,
        with_global_sos: bool = True,
        with_global_eos: bool = True,
    ):
        """将分段树合并为一个全局树，并与 seq2seq 标签对齐。

        约定：
        - 输入 forest 的每个元素均为不含本地 <sos>/<eos> 的 TreeSupervision；
        - 全局第 0 位为 <sos>（parent=-1）；
        - 仅保留前 content_len 个“内容 token”，并在末尾追加一个 <eos>（parent=-1）；
        - 返回长度为（with_global_sos ? 1 : 0）+ content_len +（with_global_eos ? 1 : 0）。
        """
        g_tokens = []
        g_parents = []
        g_rels = []

        if with_global_sos:
            g_tokens.append("<sos>")
            g_parents.append(-1)  # 根不参与结构监督
            g_rels.append("PAD")

        # 填充内容 token，累计不超过 content_len
        filled = 0
        for i, sup in enumerate(forest):
            if filled >= content_len:
                break
            tokens, parents, rels = sup.tokens, sup.parents, sup.relations
            seg_base = len(g_tokens)  # 当前 segment 在全局序列的起点
            # 遍历当前段的所有 token
            for t, p, r in zip(tokens, parents, rels):
                if filled >= content_len:
                    break
                g_tokens.append(t)
                if p == -1:
                    g_parents.append(-1)  # 保持根为忽略
                else:
                    g_parents.append(seg_base + p)  # 加 segment 偏移
                g_rels.append(r)
                filled += 1
            if insert_space and i != len(forest) - 1 and filled < content_len:
                g_tokens.append(" ")
                g_parents.append(-1)
                g_rels.append("PAD")
                filled += 1
                if filled >= content_len:
                    break

        if with_global_eos and len(g_tokens) < (1 if with_global_sos else 0) + content_len + 1:
            g_tokens.append("<eos>")
            g_parents.append(-1)
            g_rels.append("PAD")

        return g_tokens, g_parents, g_rels
