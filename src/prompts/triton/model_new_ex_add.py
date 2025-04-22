import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define the custom Triton kernel for element-wise addition
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask) 
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)

# Define the custom Triton Model for element-wise addition
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, a, b):
        out = torch.empty_like(a)
        n_elements = out.numel()
        BLOCK_SIZE = 256
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out