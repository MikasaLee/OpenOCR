from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)

# 共享解码器。
class TextIDSTreeShareDecoder(nn.Module):
    """Shared decoder producing text logits, IDS logits, and IDS structure logits."""

    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        text_vocab_path: Optional[str] = None,
        ids_vocab_path: Optional[str] = None,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 64,
        cross_coverage: bool = False,
        self_coverage: bool = False,
        max_text_length: int = 25,
        max_ids_length: int = 50,
        use_space_char: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # Strict alignment with BaseRecLabelDecode/CTCLabelEncode to avoid index shift
        def _infer_vocab_size(path: Optional[str]) -> int:
            if path is None:
                raise ValueError("vocab_path is required to infer vocab size.")
            count = 0
            with open(path, 'rb') as fin:
                lines = fin.readlines()
                # BaseRecLabelDecode reads all lines, so we count all lines
                count = len(lines)
            if use_space_char:
                count += 1
            return count + 4  # add special tokens <pad><sos><eos><unk>

        # Prioritize out_channels if provided (ensures consistency with Trainer/PostProcess)
        if out_channels is not None:
             text_vocab_size = out_channels
        else:
             text_vocab_size = _infer_vocab_size(text_vocab_path)
             
        ids_vocab_size = _infer_vocab_size(ids_vocab_path)
        self.ignore_index = 0
        self.bos_id = 1
        self.eos_id = 2
        self.max_text_len = max_text_length
        self.max_ids_len = max_ids_length
        d_model = in_channels

        self.text_embed = nn.Sequential(nn.Embedding(text_vocab_size, d_model), nn.LayerNorm(d_model))
        self.ids_embed = nn.Sequential(nn.Embedding(ids_vocab_size, d_model), nn.LayerNorm(d_model))

        # Keep positional encodings per branch to avoid unintended coupling.
        self.text_pos_enc = WordPosEnc(d_model)
        self.ids_pos_enc = WordPosEnc(d_model)

        self.text_norm = nn.LayerNorm(d_model)
        self.ids_norm = nn.LayerNorm(d_model)

        arm_factory = lambda: AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage) if (cross_coverage or self_coverage) else None
        
        # SHARED DECODER for both branches
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers,
            arm=arm_factory(),
        )
        
        self.proj_text = nn.Linear(d_model, text_vocab_size)
        self.proj_ids = nn.Linear(d_model, ids_vocab_size)
        self.struct_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=1,
        )
        self.child_proj = nn.Linear(d_model, d_model)
        self.parent_proj = nn.Linear(d_model, d_model)
        self.vs = nn.Linear(d_model, 1, bias=True)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), fill_value=1, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_memory(self, x) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        # Single-branch helper: accepts (feat2d, mask2d) or (mem, (h,w), mask_flat) or direct mem
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                feat2d, mask2d = x
                b, h, w, c = feat2d.shape
                mem = feat2d.view(b, h * w, c)
                mem_mask = mask2d.reshape(b, h * w).bool()
                return mem, h, mem_mask
            elif len(x) == 3:
                mem, hw, mem_mask = x
                h, _ = hw
                return mem, h, mem_mask.bool() if mem_mask is not None else None
        mem = x
        return mem, 2, None

    def _decode_seq(
        self,
        memory,
        h,
        mem_mask,
        tgt_ids: torch.Tensor,
        embed: nn.Module,
        proj: nn.Module,
        pos_enc: nn.Module,
        norm: nn.Module,
        decoder: TransformerDecoder,
    ):
        pad_mask = (tgt_ids == self.ignore_index)
        L = tgt_ids.size(1)
        tgt_mask = self._causal_mask(L, device=memory.device)
        tgt_emb = embed(tgt_ids)
        tgt_emb = pos_enc(tgt_emb)
        tgt_emb = norm(tgt_emb)
        # Use simple 'decoder' call which is the shared instance
        out = decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory.transpose(0, 1),
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        )
        out = out.transpose(0, 1)
        logits = proj(out)
        return logits, out, pad_mask

    def _struct_sim(self, feat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        feat = self.struct_enc(feat, src_key_padding_mask=pad_mask)
        child = self.child_proj(feat)
        parent = self.parent_proj(feat)
        M = F.relu(child.unsqueeze(2) + parent.unsqueeze(1))
        sim = self.vs(M).squeeze(-1)
        if pad_mask is not None:
            pm = pad_mask.clone()
            pm[:, 0] = True
            sim = sim.masked_fill(pm[:, None, :], float('-inf'))
        return sim

    def forward_train(self, x, data):
        # Support decoupled encoders: x can be a dict with {'text': ..., 'ids': ...}
        if isinstance(x, dict) and 'text' in x and 'ids' in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x['text'])
            mem_ids, h_ids, mem_mask_ids = self._prep_memory(x['ids'])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, h_ids, mem_mask_ids = mem_text, h_text, mem_mask_text
        text_labels = data[0]
        text_lengths = data[1]
        ids_labels = data[2]
        ids_lengths = data[3]
        # B, N, C = memory.shape

        # text branch
        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, : 2 + max_text]
        logits_text, _, pad_text = self._decode_seq(
            mem_text,
            h_text,
            mem_mask_text,
            tgt_text,
            self.text_embed,
            self.proj_text,
            self.text_pos_enc,
            self.text_norm,
            self.decoder, # Shared
        )

        # ids branch
        max_ids = int(ids_lengths.max().item())
        tgt_ids = ids_labels[:, : 2 + max_ids]
        logits_ids, hidden_ids, pad_ids = self._decode_seq(
            mem_ids,
            h_ids,
            mem_mask_ids,
            tgt_ids,
            self.ids_embed,
            self.proj_ids,
            self.ids_pos_enc,
            self.ids_norm,
            self.decoder, # Shared
        )
        sim_ids = self._struct_sim(hidden_ids, pad_ids)
        return logits_text, logits_ids, sim_ids

    def forward_test(self, x):
        if isinstance(x, dict) and 'text' in x and 'ids' in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x['text'])
            mem_ids, h_ids, mem_mask_ids = self._prep_memory(x['ids'])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, h_ids, mem_mask_ids = mem_text, h_text, mem_mask_text
        B, N, C = mem_text.shape
        device = mem_text.device

        def greedy(max_len, embed, proj, pos_enc, norm, decoder, memory, h, mem_mask):
            tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
            logits_out = []
            for i in range(max_len + 1):
                L = tgt.size(1)
                tgt_mask = self._causal_mask(L, device=device)
                pad_mask = (tgt == self.ignore_index)
                tgt_emb = embed(tgt)
                tgt_emb = pos_enc(tgt_emb)
                tgt_emb = norm(tgt_emb)
                out = decoder(
                    tgt=tgt_emb.transpose(0, 1),
                    memory=memory.transpose(0, 1),
                    height=h,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=pad_mask,
                    memory_key_padding_mask=mem_mask,
                ).transpose(0, 1)
                p_i = proj(out[:, -1:, :])
                logits_out.append(F.softmax(p_i, -1))
                if i < max_len:
                    nxt = p_i.squeeze(1).argmax(-1)
                    tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                    if (tgt == self.eos_id).any(dim=-1).all():
                        break
            return torch.cat(logits_out, dim=1)

        probs_text = greedy(
            self.max_text_len,
            self.text_embed,
            self.proj_text,
            self.text_pos_enc,
            self.text_norm,
            self.decoder, # Shared
            mem_text,
            h_text,
            mem_mask_text,
        )
        probs_ids = greedy(
            self.max_ids_len,
            self.ids_embed,
            self.proj_ids,
            self.ids_pos_enc,
            self.ids_norm,
            self.decoder, # Shared
            mem_ids,
            h_ids,
            mem_mask_ids,
        )
        return probs_text, probs_ids

    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        else:
            return self.forward_test(x)
        

if __name__ == "__main__":
    import torch
    import sys
    import os
    import shutil

    print("Initializing TextIDSTreeShareDecoder Output Check...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try to locate project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root: {project_root}")

    # Real Configs
    d_model = 512
    nhead = 4 # Reduced for speed
    num_layers = 1
    bs = 2
    max_text_len = 25
    max_ids_len = 100
    
    # Use real vocab files
    text_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    
    if not os.path.exists(text_vocab_path):
        print(f"Error: Text vocab not found at {text_vocab_path}")
        sys.exit(1)
    if not os.path.exists(ids_vocab_path):
        print(f"Error: IDS vocab not found at {ids_vocab_path}")
        sys.exit(1)

    try:
        decoder = TextIDSTreeShareDecoder(
            in_channels=d_model,
            text_vocab_path=text_vocab_path,
            ids_vocab_path=ids_vocab_path,
            nhead=nhead,
            num_decoder_layers=num_layers,
            dim_feedforward=1024,
            max_text_length=max_text_len,
            max_ids_length=max_ids_len,
            cross_coverage=False, 
            self_coverage=False,
            use_space_char=True
        ).to(device)
        
        decoder.train()
        print("Decoder initialized successfully.")
        
        # Dummy Input
        # x: [B, Seq, C]
        feat_seq_len = 32
        x = torch.randn(bs, feat_seq_len, d_model).to(device)   # [2,2*16,512]
        print(f"Input feature shape: {x.shape}")
        
        # Dummy Targets
        # Labels include special tokens: 0:pad, 1:sos, 2:eos, 3:unk. + vocab
        # text_labels: [B, L]
        # Make sure lengths imply valid range
        text_lens_val = [5, 7]
        ids_lens_val = [10, 15]
        
        # Determine strict vocab sizes for random labels
        def get_vocab_size(path, use_space):
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
            if use_space and " " not in lines:
                lines.append(" ")
            return len(lines) + 4
            
        real_text_vocab_size = get_vocab_size(text_vocab_path, True)
        real_ids_vocab_size = get_vocab_size(ids_vocab_path, True)
        
        text_labels = torch.randint(4, real_text_vocab_size, (bs, max_text_len + 5)).to(device)
        text_lens = torch.tensor(text_lens_val).to(device)
        
        ids_labels = torch.randint(4, real_ids_vocab_size, (bs, max_ids_len + 5)).to(device)
        ids_lens = torch.tensor(ids_lens_val).to(device)
        
        data = [text_labels, text_lens, ids_labels, ids_lens]
        
        print("Running forward_train...")
        logits_text, logits_ids, sim_ids = decoder(x, data)
        
        print(f"Text Logits Shape: {logits_text.shape}") 
        # Expected: [B, 2 + max(text_lens_val), VocabSize]
        
        print(f"IDS Logits Shape:  {logits_ids.shape}")  
        # Expected: [B, 2 + max(ids_lens_val), VocabSize]
        
        print(f"Struct Sim Shape:  {sim_ids.shape}")    
        # Expected: [B, 2 + max(ids_lens_val), 2 + max(ids_lens_val)]
        
        assert logits_text.shape[0] == bs
        assert logits_text.shape[1] == 2 + max(text_lens_val)
        assert logits_text.shape[2] == real_text_vocab_size, f"Text vocab size mismatch. Expected {real_text_vocab_size}, got {logits_text.shape[2]}"
        
        assert logits_ids.shape[0] == bs
        assert logits_ids.shape[1] == 2 + max(ids_lens_val)
        assert logits_ids.shape[2] == real_ids_vocab_size, f"IDS vocab size mismatch. Expected {real_ids_vocab_size}, got {logits_ids.shape[2]}"
        
        assert sim_ids.shape[0] == bs
        assert sim_ids.shape[1] == 2 + max(ids_lens_val)
        assert sim_ids.shape[2] == 2 + max(ids_lens_val)
        
        print("Shapes verification passed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with error: {e}")