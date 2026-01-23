import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class TextIDSLabelDecode:
    """Decode both text and IDS sequences (greedy output)."""

    def __init__(self, text_character_dict_path, ids_character_dict_path, char_to_ids_path=None, use_space_char=True, **kwargs):
        self.text_decoder = BaseRecLabelDecode(text_character_dict_path, use_space_char)
        self.ids_decoder = BaseRecLabelDecode(ids_character_dict_path, use_space_char)
        # align special tokens
        self.text_decoder.character = ["<pad>", "<sos>", "<eos>", "<unk>"] + self.text_decoder.character
        self.ids_decoder.character = ["<pad>", "<sos>", "<eos>", "<unk>"] + self.ids_decoder.character
        # ignore pad/sos/eos when decoding, but keep <unk> to ensure accurate matching
        self.text_decoder.get_ignored_tokens = lambda: [0, 1, 2]
        self.ids_decoder.get_ignored_tokens = lambda: [0, 1, 2]
        self.use_space_char = use_space_char

        # Build Reverse Map: IDS -> Char
        self.ids2char = {}
        if char_to_ids_path is not None:
             with open(char_to_ids_path, 'r', encoding='utf-8') as f:
                 for line in f:
                     line = line.strip()
                     if not line: continue
                     # Format: char \t ids_sequence
                     parts = line.split('\t')
                     if len(parts) >= 2:
                         char, ids_seq = parts[0], parts[1].strip()
                         self.ids2char[ids_seq] = char

    def map_ids_to_text(self, ids_seq_str):
        """Map IDS sequence string back to text string.
        Args:
            ids_seq_str: Decoded IDS string (e.g., 'ids1 ids2 ids3')
        Returns:
            Recovered text string (e.g., 'char1char2char3')
        """
        if self.use_space_char:
            segs = ids_seq_str.split(' ')
        else:
            # Fallback: if no spaces, assumed single char or ambiguous. 
            # Treating as single segment for now if use_space_char is False.
            segs = [ids_seq_str]
        
        res = []
        for s in segs:
            if not s: continue
            res.append(self.ids2char.get(s, 'X'))
        return "".join(res)

    def get_character_num(self):
        # Trainer expects a scalar; use text vocab size to align with legacy flow.
        return len(self.text_decoder.character)

    def get_vocab_sizes(self):
        # Expose both branches when needed (text, ids).
        return len(self.text_decoder.character), len(self.ids_decoder.character)

    def __call__(self, preds, batch=None, training=False, *args, **kwargs):
        # preds may be tuple (probs_text, probs_ids) or dict with 'res'
        if isinstance(preds, dict) and 'res' in preds:
            preds = preds['res']
        assert isinstance(preds, (tuple, list)) and len(preds) >= 2, "Expect tuple/list (probs_text, probs_ids)"
        probs_text, probs_ids = preds[:2]
        if isinstance(probs_text, torch.Tensor):
            probs_text = probs_text.detach().cpu().numpy()
        if isinstance(probs_ids, torch.Tensor):
            probs_ids = probs_ids.detach().cpu().numpy()

        text_dec = self._decode_branch(probs_text, self.text_decoder)
        ids_dec = self._decode_branch(probs_ids, self.ids_decoder)

        # decode labels when batch provided
        label_text = []
        label_ids = []
        if batch is not None:
            if len(batch) > 1:
                label_text = self._decode_label(batch[1], self.text_decoder)
            if len(batch) > 3:
                label_ids = self._decode_label(batch[3], self.ids_decoder)

                # Dual-branch output for RecTextIDSMetric: index0=text, index1=ids
                return [(text_dec, label_text), (ids_dec, label_ids)]

        # Inference/debug path without batch keeps dict
        # Recover text from IDS logic
        ids_recovered = []
        for (ids_str, conf) in ids_dec:
            ids_recovered.append(self.map_ids_to_text(ids_str))

        return {'text': text_dec, 'ids': ids_dec, 'text_from_ids': ids_recovered, 'label_text': label_text, 'label_ids': label_ids}

    def _decode_branch(self, probs, decoder: BaseRecLabelDecode):
        preds_idx = probs.argmax(axis=-1)
        preds_prob = probs.max(axis=-1)
        # Filter out <unk> tokens (index 3) from predictions
        preds_idx = np.where(preds_idx == 3, 0, preds_idx)  # Replace <unk> with <pad>
        
        # Use custom greedy decode with EOS truncation instead of decoder.decode
        return self._greedy_decode(preds_idx, preds_prob, decoder)

    def _greedy_decode(self, text_index, text_prob, decoder):
        """Greedy decode with EOS truncation (eos_id=2)."""
        result_list = []
        batch_size = len(text_index)
        eos_id = 2
        
        for b in range(batch_size):
            chars = []
            confs = []
            for idx, t in enumerate(text_index[b]):
                t = int(t)
                # Ignore pad(0), sos(1)
                # Note: unk(3) was already replaced by 0 in _decode_branch
                if t == 0 or t == 1:
                    continue
                # Stop at eos(2)
                if t == eos_id:
                    break
                
                if t < len(decoder.character):
                    chars.append(decoder.character[t])
                    confs.append(text_prob[b][idx])
            
            text_str = ''.join(chars)
            # Calculate average confidence of the valid sequence
            conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
            result_list.append((text_str, conf))
        return result_list

    def _decode_label(self, labels, decoder: BaseRecLabelDecode):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = labels[:, 1:]  # drop <sos>
        return decoder.decode(labels)

# python -m openrec.postprocess.text_ids_tree_postprocess
if __name__ == "__main__":
    import torch
    import os
    import sys
    import shutil
    import numpy as np
    
    # Try to locate project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root: {project_root}")
    
    # Import encoder for ground truth generation
    from openrec.preprocess.text_ids_tree_multi_label_encode import TextIDSTreeMultiLabelEncode

    print("Initializing TextIDSLabelDecode Check with REAL Configs & PREPROCESS Alignment...")
    
    # Real Configs
    max_text_len = 25
    max_ids_len = 100
    text_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")
    
    if not os.path.exists(text_dict_path):
        print(f"Error: Text vocab not found at {text_dict_path}")
        sys.exit(1)

    try:
        # 1. Initialize Encoder to get Ground Truth Indices
        print("Initializing Encoder...")
        encoder = TextIDSTreeMultiLabelEncode(
            max_text_length=max_text_len,
            max_ids_length=max_ids_len,
            character_dict_path=text_dict_path,
            ids_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True
        )

        # 2. Initialize Decoder (PostProcess)
        print("Initializing Decoder...")
        postprocess = TextIDSLabelDecode(
            text_character_dict_path=text_dict_path,
            ids_character_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True
        )
        
        text_vocab_size, ids_vocab_size = postprocess.get_vocab_sizes()
        print(f"Text Vocab Size: {text_vocab_size}")
        print(f"IDS Vocab Size: {ids_vocab_size}")
        
        # 3. Create Test Data using Encoder
        test_samples = ["我的", "你好世界", "OpenOCR", "123", "测试"]
        bs = len(test_samples)
        
        # Prepare Logits Containers
        logits_text = torch.zeros(bs, max_text_len + 2, text_vocab_size) 
        logits_ids = torch.zeros(bs, max_ids_len + 2, ids_vocab_size)
        
        # Init with low probability (simulate background)
        logits_text[:, :, 0] = 10.0 # Pad
        logits_ids[:, :, 0] = 10.0  # Pad
        
        # Prepare Labels Containers
        label_text_tensor = torch.zeros((bs, max_text_len + 2), dtype=torch.long)
        label_ids_tensor = torch.zeros((bs, max_ids_len + 2), dtype=torch.long)
        
        for i, sample_text in enumerate(test_samples):
            print(f"\n{'='*20} Processing Sample {i}: '{sample_text}' {'='*20}")
            data = {"label": sample_text}
            encoder(data) 
            
            curr_text_label = data['label']
            curr_ids_label = data['ids_label']
            
            # Fill Labels Tensor
            label_text_tensor[i, :len(curr_text_label)] = torch.from_numpy(curr_text_label)
            label_ids_tensor[i, :len(curr_ids_label)] = torch.from_numpy(curr_ids_label)
            
            # Construct Logits
            tgt_text = curr_text_label[1:] # Drop SOS
            for t, idx in enumerate(tgt_text):
                if t < logits_text.shape[1]:
                    logits_text[i, t, :] = -100.0
                    logits_text[i, t, idx] = 100.0 
            
            tgt_ids = curr_ids_label[1:] # Drop SOS
            for t, idx in enumerate(tgt_ids):
                if t < logits_ids.shape[1]:
                    logits_ids[i, t, :] = -100.0
                    logits_ids[i, t, idx] = 100.0

        print(f"\n[Model Output] Logits Prepared. Text shape: {logits_text.shape}, IDS shape: {logits_ids.shape}")
        
        probs_text = torch.softmax(logits_text, dim=-1)
        probs_ids = torch.softmax(logits_ids, dim=-1)
        
        preds = (probs_text, probs_ids)
        batch = [None, label_text_tensor, None, label_ids_tensor]
        
        print("\n[PostProcess] Running Decode...")
        res = postprocess(preds, batch)
        
        print("Decoded Result Length:", len(res))
        
        def get_item(res_list, branch, batch_idx):
            item = res_list[branch][0][batch_idx]
            if isinstance(item, tuple): item = item[0]
            return item

        print("\n[Feature Check] verifying map_ids_to_text for all samples...")
        for i, original_sample in enumerate(test_samples):
            pred_text = get_item(res, 0, i)
            pred_ids = get_item(res, 1, i)
            
            recovered = postprocess.map_ids_to_text(pred_ids)
            
            print(f"Sample {i}: '{original_sample}'")
            print(f"  > Pred Text: '{pred_text}'")
            print(f"  > Pred IDS:  '{pred_ids}'")
            print(f"  > Recovered: '{recovered}'")
            
            if pred_text: # Only check if we successfully predicted text (non-UNK)
                if recovered == pred_text:
                    print("  > Result: MATCH")
                else:
                    # It's possible some chars are not in reverse map or ambiguity
                    print(f"  > Result: MISMATCH (Expected '{pred_text}', Got '{recovered}')")
            else:
                print("  > Result: Skipped (Empty prediction due to UNK)")
            print("-" * 30)

        print("TextIDSLabelDecode Check Completed!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Check Failed: {e}")
