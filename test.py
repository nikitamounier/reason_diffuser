import torch
from accelerate import Accelerator


def func1(x):
    return x + 1


def func2(x):
    return x * 2


def func3(x):
    return x**2


def func4(x):
    return torch.sqrt(x)


def func5(x):
    return torch.log(x)


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # Create sample input tensor
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(device)

    # Run functions in parallel across GPUs
    funcs = [func1, func2, func3, func4, func5]

    results = []
    for func in funcs:
        with accelerator.split_between_processes():
            result = func(x)
            results.append(result)

    # Gather results
    results = accelerator.gather(results)

    if accelerator.is_main_process:
        print("Results:", results)


if __name__ == "__main__":
    main()
