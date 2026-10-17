"""A runnable introduction to tensors, logits, cross entropy and autograd."""
import torch
from torch import nn


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    centers = torch.tensor([[-2., -2.], [2., -2.], [0., 2.]])
    y = torch.arange(3).repeat_interleave(40)
    x = centers[y] + 0.4 * torch.randn(120, 2)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
    for step in range(101):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step={step:3d} loss={loss.item():.4f}")
    with torch.inference_mode():
        print("logits shape:", tuple(model(x).shape))
        print("toy TRAIN accuracy:", (model(x).argmax(1) == y).float().mean().item())
        print("Only an autograd exercise; this toy training accuracy is not a project result.")


if __name__ == "__main__":
    main()
