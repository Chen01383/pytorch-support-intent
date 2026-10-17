"""A small, understandable PyTorch classifier trained on frozen embeddings."""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


class IntentHead(nn.Module):
    def __init__(self, input_dim=384, num_classes=77, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, num_classes)) if hidden_dim else nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return self.network(embeddings)


def train_head(features, targets, hidden_dim, seed, plan, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = IntentHead(features["train"].shape[1], len(set(targets["train"])), hidden_dim, plan["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    dataset = TensorDataset(torch.from_numpy(features["train"]), torch.tensor(targets["train"], dtype=torch.long))
    loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    selection_x = torch.from_numpy(features["selection"]).to(device)
    best_score, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, plan["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        model.eval()
        with torch.inference_mode():
            selection_logits = model(selection_x).cpu().numpy()
        score = float(f1_score(targets["selection"], selection_logits.argmax(1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_cross_entropy": total_loss / len(dataset), "selection_macro_f1": score})
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= plan["patience"]:
            break
    model.load_state_dict(best_state)
    model.cpu().eval()
    print(f"hidden={hidden_dim}, seed={seed}, best_epoch={best_epoch}, selection_macro_f1={best_score:.5f}", flush=True)
    return model, {"seed": seed, "hidden_dim": hidden_dim, "best_epoch": best_epoch,
                   "selection_macro_f1": best_score, "history": history}
