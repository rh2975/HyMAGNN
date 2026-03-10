
import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from util import make_loaders, mae, rmse, rse, corr
from magnn import MAGNN



def evaluate(model, loader, scaler, device, use_amp=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)          # (B,1,N,T)
            y = y.to(device, non_blocking=True)          # (B,H,N,1)

            with autocast(enabled=use_amp):
                out, _ = model(x)                        # (B,H,N,1)

            out_np = out.squeeze(-1).cpu().numpy()       # (B,H,N)
            y_np   = y.squeeze(-1).cpu().numpy()         # (B,H,N)

            out_np = scaler.inverse_transform(out_np.reshape(-1, out_np.shape[-1])).reshape(out_np.shape)
            y_np   = scaler.inverse_transform(y_np.reshape(-1, y_np.shape[-1])).reshape(y_np.shape)

            preds.append(out_np)
            trues.append(y_np)

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    return {
        "MAE": mae(pred, true),
        "RMSE": rmse(pred, true),
        "RSE": rse(pred, true),
        "CORR": corr(pred, true),
    }


def benchmark_inference_time(model, loader, device, use_amp=False,
                             warmup_batches=5, timed_batches=30):
    """
    Measures forward-pass time only (data already on GPU).
    Reports:
      - avg ms/batch
      - avg ms/sample
      - throughput samples/sec
    """
    model.eval()

    # pick how many batches we can actually use
    total_batches = 0
    for _ in loader:
        total_batches += 1
        if total_batches >= (warmup_batches + timed_batches):
            break

    timed_batches = max(1, min(timed_batches, max(0, total_batches - warmup_batches)))

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

    times_ms = []
    n_samples = 0

    with torch.no_grad():
        b = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            bs = x.size(0)

            # warmup
            if b < warmup_batches:
                with autocast(enabled=use_amp):
                    _ = model(x)
                b += 1
                continue

            if len(times_ms) >= timed_batches:
                break

            if device.type == "cuda":
                torch.cuda.synchronize()
                starter.record()
                with autocast(enabled=use_amp):
                    _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)  # ms
            else:
                t0 = time.perf_counter()
                with autocast(enabled=use_amp):
                    _ = model(x)
                elapsed = (time.perf_counter() - t0) * 1000.0

            times_ms.append(elapsed)
            n_samples += bs
            b += 1

    avg_ms_batch = float(np.mean(times_ms))
    avg_ms_sample = avg_ms_batch / (n_samples / len(times_ms))
    throughput = 1000.0 / avg_ms_sample  # samples/sec

    return {
        "avg_ms_per_batch": avg_ms_batch,
        "avg_ms_per_sample": avg_ms_sample,
        "throughput_samples_per_sec": throughput,
        "timed_batches": len(times_ms),
        "warmup_batches": warmup_batches,
    }


def build_model(args, num_nodes):
    return MAGNN(
        num_nodes=num_nodes,
        seq_in_len=args.seq_in_len,
        horizon=args.horizon,
        num_scales=4,
        topk=args.topk,
        embed_dim=args.embed_dim,
        conv_channels=args.conv_channels,
        ds=args.ds,
        gcn_depth=args.gcn_depth,
        dropout=args.dropout,
        alpha=args.propalpha,
        use_hypergraph=args.use_hypergraph,
        num_hyperedges=args.num_hyperedges,
        use_deform_conv=args.use_deform_conv,
        use_core_fusion=args.use_core_fusion,
        core_hidden_dim=args.core_hidden_dim
    ).to(args.device)


def train(args):
    train_loader, val_loader, test_loader, scaler, num_nodes = make_loaders(
        args.data_path, args.seq_in_len, args.horizon,
        args.batch_size, num_workers=args.num_workers,
        split=(0.6,0.2,0.2)
    )

    model = build_model(args, num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler(enabled=args.amp)

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                out, _ = model(x)
                loss = criterion(out, y)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()

        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, scaler, args.device, use_amp=args.amp)
            val_loss_proxy = val_metrics["RMSE"]

            print(f"Epoch {epoch:03d} | TrainLoss {total_loss/len(train_loader):.6f} | "
                  f"VAL MAE {val_metrics['MAE']:.4f} RMSE {val_metrics['RMSE']:.4f} "
                  f"RSE {val_metrics['RSE']:.4f} CORR {val_metrics['CORR']:.4f}")

            if val_loss_proxy < best_val:
                best_val = val_loss_proxy
                torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
                print("  -> Saved best:", ckpt_path)

    print("\nLoading best checkpoint for TEST + Inference Time...")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, scaler, args.device, use_amp=args.amp)
    print(f"TEST  MAE {test_metrics['MAE']:.4f} RMSE {test_metrics['RMSE']:.4f} "
          f"RSE {test_metrics['RSE']:.4f} CORR {test_metrics['CORR']:.4f}")

    # inference time report (forward-only)
    tstat = benchmark_inference_time(
        model, test_loader, args.device,
        use_amp=args.amp,
        warmup_batches=5,
        timed_batches=30
    )
    print(f"INFER_TIME (forward-only) | {tstat['avg_ms_per_batch']:.3f} ms/batch | "
          f"{tstat['avg_ms_per_sample']:.3f} ms/sample | "
          f"{tstat['throughput_samples_per_sec']:.2f} samples/s | "
          f"(timed {tstat['timed_batches']} batches, warmup {tstat['warmup_batches']})")


def inference(args):
    _, _, test_loader, scaler, num_nodes = make_loaders(
        args.data_path, args.seq_in_len, args.horizon,
        args.batch_size, num_workers=args.num_workers,
        split=(0.6,0.2,0.2)
    )

    model = build_model(args, num_nodes)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate(model, test_loader, scaler, args.device, use_amp=args.amp)
    print("INFER metrics:", metrics)

    #inference time report (forward-only)
    tstat = benchmark_inference_time(
        model, test_loader, args.device,
        use_amp=args.amp,
        warmup_batches=5,
        timed_batches=30
    )
    print(f"INFER_TIME (forward-only) | {tstat['avg_ms_per_batch']:.3f} ms/batch | "
          f"{tstat['avg_ms_per_sample']:.3f} ms/sample | "
          f"{tstat['throughput_samples_per_sec']:.2f} samples/s | "
          f"(timed {tstat['timed_batches']} batches, warmup {tstat['warmup_batches']})")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_name", type=str, default="magnn_best.pt")
    p.add_argument("--ckpt", type=str, default="")

    p.add_argument("--seq_in_len", type=int, default=168)
    p.add_argument("--horizon", type=int, default=3)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--embed_dim", type=int, default=40)
    p.add_argument("--conv_channels", type=int, default=32)
    p.add_argument("--ds", type=int, default=32)
    p.add_argument("--gcn_depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--propalpha", type=float, default=0.05)

    p.add_argument("--use_hypergraph", action="store_true")
    p.add_argument("--num_hyperedges", type=int, default=32)

    p.add_argument("--use_deform_conv", action="store_true")
    p.add_argument("--use_core_fusion", action="store_true")
    p.add_argument("--core_hidden_dim", type=int, default=64)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--mode", type=str, choices=["train","infer"], default="train")

    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt:
            raise ValueError("--ckpt required for infer mode")
        inference(args)

if __name__ == "__main__":
    main()
