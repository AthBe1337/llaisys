import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_linear(out, x, w, bias):
    if out.dtype == torch.int8:
        x_f = x.to(torch.float32)
        w_f = w.to(torch.float32)
        b_f = bias.to(torch.float32) if bias is not None else None
        
        res = torch.nn.functional.linear(x_f, w_f, b_f)
        
        res = res.round().clamp(-128, 127)
        
        out.copy_(res.to(torch.int8))
    else:
        torch.nn.functional.linear(x, w, bias, out=out)


def test_op_linear(
    out_shape,
    x_shape,
    w_shape,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   out {out_shape}, x {x_shape}, w {w_shape}, bias {use_bias}, dtype <{dtype_name}>"
    )
    if dtype_name not in ["i8"]:
        x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
        w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)
    else:
        x, x_ = random_int_tensor(x_shape, device_name, dtype_name,low=-128, high=127)
        w, w_ = random_int_tensor(w_shape, device_name, dtype_name, low=-128, high=127)

    bias, bias_ = None, None
    if use_bias:
        if dtype_name not in ["i8"]:
            bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)
        else:
            bias, bias_ = random_int_tensor(
                (w_shape[0],), device_name, dtype_name, low=-128, high=127
            )
    if dtype_name not in ["i8"]:
        out, out_ = random_tensor(out_shape, dtype_name, device_name)
    else:
        out, out_ = random_int_tensor(out_shape, device_name, dtype_name, low=-128, high=127)

    torch_linear(out, x, w, bias)
    llaisys.Ops.linear(out_, x_, w_, bias_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_linear(out, x, w, bias),
            lambda: llaisys.Ops.linear(out_, x_, w_, bias_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        # ((1, 1536), (1, 1536), (1536, 1536), True),
        # ((1, 8960), (1, 1536), (8960, 1536), True),
        # ((1, 1536), (1, 8960), (1536, 8960), True),
        ((128, 1536), (128, 1536), (1536, 1536), True),
        ((128, 8960), (128, 1536), (8960, 1536), True),
        ((128, 1536), (128, 8960), (1536, 8960), True),
        # ((1, 4096), (1, 4096), (4096, 4096), True),
        # ((1, 12288), (1, 4096), (12288, 4096), True),
        # ((1, 4096), (1, 12288), (4096, 12288), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
        ((512, 12288), (512, 4096), (12288, 4096), True),
        ((512, 4096), (512, 12288), (4096, 12288), True),
        # ((1, 5120), (1, 5120), (5120, 5120), True),
        # ((1, 27648), (1, 5120), (27648, 5120), True),
        # ((1, 5120), (1, 27648), (5120, 27648), True),
        ((1024, 5120), (1024, 5120), (5120, 5120), True),
        ((1024, 27648), (1024, 5120), (27648, 5120), True),
        ((1024, 5120), (1024, 27648), (5120, 27648), True),
    ]
    testDtypePrec = [
        # type, atol, rtol
        # ("f32", 5e-5, 5e-5),
        # ("f16", 1e-3, 1e-3),
        ("bf16", 5e-2, 5e-2),
        ("i8", 0, 0),
    ]
    print(f"Testing Ops.linear on {args.device}")
    for shapes in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_linear(*shapes, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
