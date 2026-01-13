import torch
import timeit
from tqdm import tqdm
from cs336_basics.model import BasicsTransformerLM

def benchmarking(
    num_layers: int,
    num_heads: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    seq_length: int,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
    backward: bool = True,
):
    """
    Basic end-to-end benchmarking of the forward and backward passes in your model. Should support the following:
    - Given hyperparameters (e.g., number of layers), initialize a model.
    - Generate a random batch of data.
    - Run w warm-up steps (before you start measuring time), then time the execution of n steps (either only forward, or both forward and backward passes, depending on an argument). For timing, you can use the Python timeit module (e.g., either using the timeit function, or using timeit.default_timer(), which gives you the system’s highest resolution clock, thus a better default for benchmarking than time.time()).
    - Call torch.cuda.synchronize() after each step.
    Deliverable: A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward and backward passes.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = BasicsTransformerLM(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=seq_length,
        rope_theta=10000,
    ).to(device)

    # Generate random batch of data
    input_data = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    target_data = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

    # 单步：前向传播，计算loss，可选反向传播
    # 每个单步结束后强制同步
    def step():
        outputs = model(input_data)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, vocab_size), target_data.view(-1)
        )
        if backward:
            loss.backward()
        torch.cuda.synchronize()

    # Warm-up steps
    for _ in tqdm(range(warmup_steps),desc="Warm-up"):
        step()

    # Timed steps
    start_time = timeit.default_timer()
    for _ in tqdm(range(timed_steps),desc="Timed Steps"):
        step()
    end_time = timeit.default_timer()

    total_time = end_time - start_time
    avg_time_per_step = total_time / timed_steps

    print(f"Total time for {timed_steps} steps: {total_time:.4f} seconds")
    print(f"Average time per step: {avg_time_per_step:.4f} seconds")

    return {
        "timed_steps": timed_steps,
        "total_time": total_time,
        "avg_time_per_step": avg_time_per_step,
    }

def main():
    print("Benchmarking Basics Transformer LM")
    benchmarking(
        num_layers=12,
        num_heads=12,
        d_model=768,
        d_ff=3072,
        vocab_size=10000,
        seq_length=256,
        batch_size=4,
        warmup_steps=5,
        timed_steps=10,
        backward=True,
    )
    print("Benchmarking completed.")

if __name__ == "__main__":
    main()