import os
import random
import time

import numpy as np
import torch


def enable_reproducibility(
        seed=0, raise_if_no_deterministic=True,
        cudnn_deterministic=True, disable_cudnn_benchmarking=True):
    # https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    torch.use_deterministic_algorithms(raise_if_no_deterministic)

    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    torch.backends.cudnn.benchmark = not disable_cudnn_benchmarking
    torch.backends.cudnn.deterministic = cudnn_deterministic

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import imgaug as ia
        ia.seed(seed)
    except ImportError:
        pass


def try_to_clear_gpu_memory(gpu_related_variables=("model", "optimizer", "criterion", "scheduler")):
    for name in gpu_related_variables:
        try:
            del globals()[name]
        except KeyError:
            pass
    torch.cuda.empty_cache()


def count_boxes_overshooted_edges(boxes_cx_cy_h_w):
    cx, cy, w, h = boxes_cx_cy_h_w.unbind(-1)
    overshooted_left_edge = (cx - w / 2) < 0
    overshooted_right_edge = (cx + w / 2) > 1
    overshooted_top_edge = (cy - h / 2) < 0
    overshooted_bottom_edge = (cy + h / 2) > 1
    overshooted = torch.stack(
        (
            overshooted_left_edge,
            overshooted_right_edge,
            overshooted_top_edge,
            overshooted_bottom_edge
        ),
        dim=-1)
    return overshooted.any(dim=-1).sum().item()


@torch.no_grad()
def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32',
              nwarmup=50, nruns=1000, device='cuda'):
    was_training = model.training
    model.eval()
    device = torch.device(device)
    if device.type == torch.device('cuda').type:
        was_in_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True

    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    if nwarmup > 0:
        print("Warm up ...")
    for _ in range(nwarmup):
        model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    for i in range(1, nruns + 1):
        start_time = time.time()
        pred_loc, pred_label, *_ = model(input_data)
        torch.cuda.synchronize()
        end_time = time.time()
        timings.append(end_time - start_time)
        if i % 100 == 0:
            print("Iteration {:d}/{:d}, avg batch time {:.2f} ms".format(
                  i, nruns, torch.tensor(timings).mean() * 1000))

    model.train(was_training)
    if device.type == torch.device('cuda').type:
        torch.backends.cudnn.benchmark = was_in_benchmark

    print("Input shape:", input_data.size())
    print("Output location prediction size:", pred_loc.size())
    print("Output label prediction size:", pred_label.size())
    print("Average batch time: {:.2f} ms".format(
        torch.tensor(timings).mean() * 1000))

# benchmark(model, input_shape=(1, 3, 224, 224), nwarmup=0, nruns=1, device='cpu')
