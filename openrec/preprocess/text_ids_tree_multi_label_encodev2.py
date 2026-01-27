import numpy as np

from tools.utils.ids_to_tree import (
    ids_to_tree_supervision,
    load_char_to_ids,
)
from .ctc_label_encode import BaseRecLabelEncode


def _load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip("\n\r") for ln in f if ln.strip("\n\r")]


class TextIDSTreeMultiLabelEncodev2(BaseRecLabelEncode):
    """
    输出（训练建议）:
      - label:        text token ids, shape [2 + max_text_len]
      - length:       text content length (no sos/eos)
      - label_ids_seq:ids token ids,  shape [2 + max_ids_len]
      - length_ids:   ids content length (no sos/eos)
      - tree_tokens:  ids token ids aligned with label_ids_seq (推荐直接复用)
      - tree_parents: parent pointers aligned with label_ids_seq
    特殊符号约定：<pad>=0,<sos>=1,<eos>=2,<unk>=3
    """

    def __init__(
        self,
        max_text_length,
        max_ids_length,
        character_dict_path,
        ids_dict_path,
        char_to_ids_path,
        use_space_char=False,
        **kwargs,
    ):
        super().__init__(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            lower=kwargs.get("lower", False),
        )

        # --- text vocab (来自 BaseRecLabelEncode)
        self.text_dict = self.dict
        self.text_max_len = max_text_length
        self.text_PAD = self.text_dict["<pad>"]
        self.text_SOS = self.text_dict["<sos>"]
        self.text_EOS = self.text_dict["<eos>"]
        self.text_UNK = self.text_dict["<unk>"]

        # --- ids vocab (独立)
        ids_chars = _load_list(ids_dict_path)
        if use_space_char and " " not in ids_chars:
            ids_chars.append(" ")

        # 保证 ids vocab 也有同样 special token index 约定
        ids_chars = ["<pad>", "<sos>", "<eos>", "<unk>"] + ids_chars
        self.ids_dict = {ch: i for i, ch in enumerate(ids_chars)}
        self.ids_max_len = max_ids_length
        self.ids_PAD = self.ids_dict["<pad>"]
        self.ids_SOS = self.ids_dict["<sos>"]
        self.ids_EOS = self.ids_dict["<eos>"]
        self.ids_UNK = self.ids_dict["<unk>"]

        self.use_space_char = use_space_char

        # char -> IDS string
        self.char2ids = load_char_to_ids(char_to_ids_path)

    def add_special_char(self, dict_character):
        # 覆盖 BaseRecLabelEncode 的 special token 顺序，确保索引对齐
        return ["<pad>", "<sos>", "<eos>", "<unk>"] + dict_character

    def __call__(self, data):
        text = data["label"]
        if self.lower:
            text = text.lower()

        # ========== 1) 文本分支：text -> label ==========
        # 不要丢字符：未知字符映射到 <unk>
        text_tokens = list(text)
        text_ids = [
            self.text_dict.get(ch, self.text_UNK) for ch in text_tokens
        ]
        text_len = min(len(text_ids), self.text_max_len)
        label_text = [self.text_SOS] + text_ids[:text_len] + [self.text_EOS]
        total_text = 2 + self.text_max_len
        if len(label_text) < total_text:
            label_text += [self.text_PAD] * (total_text - len(label_text))

        data["label"] = np.array(label_text, dtype=np.int64)   # 文本序列
        data["length"] = np.array(text_len, dtype=np.int64)    # 文本长度（不含 sos/eos）

        # ========== 2) IDS 分支：text -> IDS string -> label_ids_seq ==========

        # 每个字符一个 IDS segment，用于建 forest（结构监督）
        ids_segments = []
        for ch in text_tokens[:text_len]:
            ids_str = self.char2ids.get(ch)
            if ids_str is None:
                # 兜底：如果字符本身就在 ids vocab，就用它；否则用 <unk> 占位
                ids_str = ch if ch in self.ids_dict else "<unk>"
            ids_segments.append(ids_str)

        # 拼成线性 IDS token 序列（用于 token supervision）
        ids_linear_tokens = []
        for i, seg in enumerate(ids_segments):
            # seg 是字符串，按字符粒度切 token（IDC/部件都是单字符的常见场景）
            # [Fix] 如果 seg 是 <unk> 等特殊 token (如字典缺失兜底产生的)，当做一个整体，不用 list() 拆散
            if seg.startswith("<") and seg.endswith(">"):
                ids_linear_tokens.append(seg)
            else:
                ids_linear_tokens.extend(list(seg))

            if self.use_space_char and i != len(ids_segments) - 1:
                ids_linear_tokens.append(" ")

        ids_linear_ids = [
            self.ids_dict.get(tok, self.ids_UNK) for tok in ids_linear_tokens
        ]
        ids_len = min(len(ids_linear_ids), self.ids_max_len)

        label_ids_seq = [self.ids_SOS] + ids_linear_ids[:ids_len] + [self.ids_EOS]
        total_ids = 2 + self.ids_max_len
        if len(label_ids_seq) < total_ids:
            label_ids_seq += [self.ids_PAD] * (total_ids - len(label_ids_seq))

        data["ids_label"] = np.array(label_ids_seq, dtype=np.int64)   # IDS 序列
        data["ids_length"] = np.array(ids_len, dtype=np.int64)        # IDS 长度（不含 sos/eos）

        # CTC 需要的 label（不含 sos/eos，pad 到 max_ids_len）
        ids_ctc = ids_linear_ids[:ids_len]
        ids_ctc_padded = ids_ctc + [self.ids_PAD] * (self.ids_max_len - len(ids_ctc))
        data["ids_ctc_label"] = np.array(ids_ctc_padded, dtype=np.int64)
        data["ids_ctc_length"] = np.array(ids_len, dtype=np.int64)
        # ========== 3) 结构分支：IDS segments -> forest -> global tree (对齐 IDS) ==========
        forest = []
        for seg in ids_segments:
            # 注意：ids_to_tree_supervision 需要的是“单字 IDS”，不要带你插入的空格
            forest.append(ids_to_tree_supervision(seg, add_sos=False, add_eos=False))

        tree_tokens_str, tree_parents = self._build_global_tree(
            forest=forest,
            content_len=ids_len,
            insert_space=self.use_space_char,
        )

        # tree_tokens 建议输出 int，并与 label_ids_seq 对齐（最稳）
        # 这里把 tree_tokens_str 映射成 ids vocab id；再 pad 到 total_ids
        tree_tokens = [self.ids_SOS]
        for tok in tree_tokens_str[1:-1]:  # 去掉 build_global_tree 自带的 <sos>/<eos>
            tree_tokens.append(self.ids_dict.get(tok, self.ids_UNK))
        tree_tokens.append(self.ids_EOS)

        # pad
        if len(tree_tokens) < total_ids:
            tree_tokens += [self.ids_PAD] * (total_ids - len(tree_tokens))
        if len(tree_parents) < total_ids:
            tree_parents += [-1] * (total_ids - len(tree_parents))

        # 结构父指针与 ids_label 对齐（长度 = total_ids）
        data["tree_parents_label"] = np.array(tree_parents, dtype=np.int64)

        # Debug 可读字段（不进 KeepKeys 即不会传到模型）
        data["text_token"] = text_tokens[:text_len]
        data["ids_token"] = ids_linear_tokens[:ids_len]
        data["tree_tokens"] = tree_tokens_str  # 可读字符串列表


        # tree_ctc_tokens 建议输出 int，并与 ids_ctc_label 对齐（最稳）
        # # CTC 需要的 label（不含 sos/eos，pad 到 max_ids_len）
        tree_ctc_tokens = []
        for tok in tree_tokens_str[1:-1]:  # 去掉 build_global_tree 自带的 <sos>/<eos>
            tree_ctc_tokens.append(self.ids_dict.get(tok, self.ids_UNK))
        tree_parents_ctc = []
        for p in tree_parents[1:-1]:  # drop <sos>/<eos> positions
            if p <= 0:
                tree_parents_ctc.append(-1)
            else:
                tree_parents_ctc.append(p - 1)  # ✅ shift: drop BOS index

        # pad
        if len(tree_ctc_tokens) < self.ids_max_len:
            tree_ctc_tokens += [self.ids_PAD] * (self.ids_max_len - len(tree_ctc_tokens))
        if len(tree_parents_ctc) < self.ids_max_len:
            tree_parents_ctc += [-1] * (self.ids_max_len - len(tree_parents_ctc))

        # 结构父指针与 ids_ctc_label 对齐（长度 = self.ids_max_len
        data["tree_parents_ctc_label"] = np.array(tree_parents_ctc, dtype=np.int64)

        # Debug 可读字段（不进 KeepKeys 即不会传到模型）
        data["text_ctc_token"] = text_tokens[:text_len]
        data["ids_ctc_token"] = ids_linear_tokens[:ids_len]
        data["tree_ctc_tokens"] = tree_tokens_str[1:-1] # 去掉 build_global_tree 自带的 <sos>/<eos>

        # print(f'data:{data}')  # debug
        return data

    def _build_global_tree(self, forest, content_len, insert_space=False):
        """
        返回：
          tokens:   ["<sos>"] + 内容token[:content_len] + ["<eos>"]
          parents:  与 tokens 等长；根/空格/eos 的 parent = -1
        """
        tokens = ["<sos>"]
        parents = [-1]

        filled = 0
        for i, sup in enumerate(forest):
            if filled >= content_len:
                break
            seg_base = len(tokens)
            for t, p in zip(sup.tokens, sup.parents):
                if filled >= content_len:
                    break
                tokens.append(t)
                parents.append(-1 if p == -1 else seg_base + p)
                filled += 1

            if insert_space and i != len(forest) - 1 and filled < content_len:
                tokens.append(" ")
                parents.append(-1)
                filled += 1

        tokens.append("<eos>")
        parents.append(-1)
        return tokens, parents
    
# python -m openrec.preprocess.text_ids_tree_multi_label_encodev2
if __name__ == "__main__":
    import os
    import sys

    # Try to locate project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Project root set to: {project_root}")

    # Configs
    max_text_len = 15
    max_ids_len = 100
    
    char_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")

    print(f"Loading encoder...")
    try:
        encoder = TextIDSTreeMultiLabelEncodev2(
            max_text_length=max_text_len,
            max_ids_length=max_ids_len,
            character_dict_path=char_dict_path,
            ids_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True 
        )
        print("Encoder loaded.")
    except Exception as e:
        print(f"Failed to load encoder: {e}")
        if not os.path.exists(char_dict_path):
             print(f"File not found: {char_dict_path}")
        sys.exit(1)

    # Test samples
    test_labels = ["一到春天，", "我", "我的", "test", "赢", "aaaa"] 
    for label in test_labels:
        data = {"label": label}
        print(f"\n--- Testing label: {label} ---")
        try:
            data_copy = data.copy()
            encoder(data_copy)
            
            print(f"Text Label IDs: {data_copy['label']}")
            print(f"IDS Label IDs: {data_copy['ids_ctc_label']}")
            print(f"IDS Length: {data_copy['ids_ctc_length']}")
            print(f"Tree Parents Label: {data_copy['tree_parents_ctc_label']}")
            
            inv_ids_dict = {v: k for k, v in encoder.ids_dict.items()}
            decoded_ids = [inv_ids_dict.get(idx, f"<{idx}>") for idx in data_copy["ids_ctc_label"] if idx != 0]
            print(f"Decoded IDS Sequence: {''.join(decoded_ids)}")

            unk_cnt = np.sum(data_copy["ids_ctc_label"] == encoder.ids_UNK)
            print(f"UNK count in IDS: {unk_cnt}")

            # 可读的树 token 与父指针（截取有效长度部分）
            tree_tokens_str = data_copy.get("tree_ctc_tokens", [])
            parents_full = data_copy["tree_parents_ctc_label"].tolist()
            valid_len = int(data_copy["ids_ctc_length"])
            tokens_trim = tree_tokens_str[:valid_len]
            parents_trim = parents_full[:valid_len]
            print("Tree tokens (trimmed):", tokens_trim)
            print("Tree parents (trimmed):", parents_trim)
            print("Tree idx->token mapping (trimmed):")
            for i, (tok, p) in enumerate(zip(tokens_trim, parents_trim)):
                print(f"  idx={i:02d} tok={tok} parent={p}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {label}: {e}")
