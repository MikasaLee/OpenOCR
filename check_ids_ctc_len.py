# check_ids_ctc_len.py
import argparse
import logging
import numpy as np
import torch

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer

def make_logger():
    logger = logging.getLogger("check_ids_ctc_len")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger

def infer_time_steps(enc_out, d_model_hint=None):
    """
    尽量从 encoder 输出里推断 CTC 的时间步 T。
    返回 (T, shape_str) 方便你打印确认。
    """
    x = enc_out
    # 常见：tuple/list
    if isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    # 常见：dict
    if isinstance(x, dict):
        for k in ["ids", "text", "res", "feat", "features", "out", "x"]:
            if k in x:
                x = x[k]
                break
    # 再解一层
    if isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]

    if not torch.is_tensor(x):
        return None, f"type={type(x)}"

    shp = tuple(x.shape)
    if x.ndim == 3:
        # [B, T, C]
        return int(shp[1]), f"3D {shp} (assume [B,T,C])"
    if x.ndim == 4:
        # 可能是 [B,H,W,C] 或 [B,C,H,W]
        B = shp[0]
        # 猜测 C 维
        if d_model_hint is not None:
            if shp[-1] == d_model_hint:
                H, W = shp[1], shp[2]
                return int(H * W), f"4D {shp} (assume [B,H,W,C], T=H*W)"
            if shp[1] == d_model_hint:
                H, W = shp[2], shp[3]
                return int(H * W), f"4D {shp} (assume [B,C,H,W], T=H*W)"
        # 兜底：取 H*W 两种都打印，让你看
        return None, f"4D {shp} (unknown layout)"
    return None, f"{x.ndim}D {shp}"

def main(cfg_path, mode="Train", num_batches=200, device="cuda"):
    logger = make_logger()

    cfg = Config(cfg_path)
    # 建议：统计脚本别走分布式
    cfg.cfg["Global"]["distributed"] = False
    cfg.cfg["Global"]["use_amp"] = False

    # 用 Trainer 拿到 model + logger（避免你遇到的 logger=None）
    trainer = Trainer(cfg, mode="eval")
    model = trainer.model
    model.eval()
    dev = trainer.device if hasattr(trainer, "device") else torch.device(device)

    dl = build_dataloader(cfg.cfg, mode, logger)
    logger.info(f"Loaded dataloader mode={mode}, iters={len(dl)}")

    ids_lens = []
    Ts = []
    bad = 0
    total = 0

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= num_batches:
                break

            # keep_keys: [0:image, 1:label, 2:length, 3:ids_ctc_label, 4:ids_ctc_length, 5:tree_parents_label]
            ids_len = batch[4]
            if torch.is_tensor(ids_len):
                ids_len_np = ids_len.cpu().numpy().astype(np.int32).reshape(-1)
            else:
                ids_len_np = np.asarray(ids_len).astype(np.int32).reshape(-1)
            ids_lens.extend(ids_len_np.tolist())

            # encoder 输出时间步 T
            imgs = batch[0].to(dev)
            if hasattr(model, "encoder"):
                enc_out = model.encoder(imgs)
            elif hasattr(model, "backbone"):
                enc_out = model.backbone(imgs)
            else:
                raise RuntimeError("Cannot find model.encoder or model.backbone to compute time steps")

            T, shape_str = infer_time_steps(enc_out)
            if T is not None:
                Ts.append(T)
                total += int(ids_len_np.shape[0])
                bad += int((ids_len_np > T).sum())
            else:
                logger.warning(f"Cannot infer T from encoder output: {shape_str}")

            if bi == 0:
                logger.info(f"[Sanity] encoder_out: {shape_str}")
                logger.info(f"[Sanity] example ids_ctc_length: {ids_len_np[:8].tolist()}")

    ids_lens = np.array(ids_lens, dtype=np.int32)
    logger.info(f"Collected ids_ctc_length: n={len(ids_lens)}")

    def q(arr):
        return {
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }

    logger.info(f"ids_ctc_length quantiles: {q(ids_lens)}")

    if len(Ts) > 0:
        Ts = np.array(Ts, dtype=np.int32)
        logger.info(f"CTC time steps T (encoder output len) quantiles over batches: {q(Ts)}")
        logger.info(f"Mismatch rate: ids_ctc_length > T  =>  {bad}/{total} = {bad/ max(total,1):.4%}")
        # 额外给一个“需要的 out_char_num 下限”建议：至少 >= ids_len p99
        logger.info(f"Suggestion: set CTC T >= ids_ctc_length p99 (= {np.quantile(ids_lens, 0.99):.1f}) (ideally +10~20% margin)")
    else:
        logger.warning("No valid T inferred. Check encoder output format.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="Train", choices=["Train", "Eval"])
    ap.add_argument("--num_batches", type=int, default=200)
    args = ap.parse_args()
    main(args.config, args.mode, args.num_batches)
