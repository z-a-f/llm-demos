import torch

def register_lora_hook_all_linear(model, rank):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            register_lora_hook(module, rank)

def register_lora_hook(module, rank):
    '''Registers a forward posthook for linear layers.
    
    The post-hook should be equivalent to the original definition of LoRA.
    Note that LoRA is defined as $W' = W + AB$, where $A$ and $B$ are learnable.
    The post hook is applied as $h(X @ W + b)$.
    We can define $h$ as $h(X @ W + b) = X @ (W + AB) + b = X @ W + b + X @ AB$.
    '''
    def _lora_hook(mod, args, output):
        # mod: Linear layer
        # X: Inputs that module received during forward pass
        # output: The output of the module after the forward pass
        A = module.residual_a  # A is Cout x rank
        B = module.residual_b  # B is rank x Cin
        X = args[0]
        output += X @ (A @ B).T
        return output
    Cout = module.weight.shape[0]
    Cin = module.weight.shape[1]
    device = module.weight.device
    dtype = module.weight.dtype
    module.register_parameter(
        'residual_a',
        torch.nn.Parameter(torch.randn(Cout, rank, device=device, dtype=dtype) * 1e-2)
    )
    module.register_parameter(
        'residual_b',
        torch.nn.Parameter(torch.randn(rank, Cin, device=device, dtype=dtype) * 1e-2)
    )
    module.register_forward_hook(_lora_hook, prepend=True)