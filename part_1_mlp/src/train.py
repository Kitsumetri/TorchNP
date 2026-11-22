"""
def train_one_epoch_torch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        optimizer.zero_grad()
        logits = model(xb) # forward на батч
        loss = criterion(logits, yb) # MSE на батч
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        # суммируем где сошлись аутпуты модели с oh меткой
        total_correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")
    return total_loss / total, total_correct / total

@torch.inference_mode()
def evaluate_torch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        # суммируем где сошлись аутпуты модели с oh меткой
        total_correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
        total += bs
    return total_loss / total, total_correct / total
"""

# За основу брал свою же реализацию на торче

from tqdm.auto import tqdm

def train_one_epoch_np(model, loader, criterion, optimizer):
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        optimizer.zero_grad()
        logits = model(xb) # forward на батч
        loss_val, dY = criterion(logits, yb) # loss и dL/dY_pred
        model.backward(dY) # backprop руками из модели
        optimizer.step()

        bs = xb.shape[0]
        total_loss += loss_val * bs
        total_correct += int((logits.argmax(axis=1) == yb.argmax(axis=1)).sum())
        total += bs
    return total_loss / total, total_correct / total


def evaluate_np(model, loader, criterion):
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        logits = model(xb)
        loss_val, _ = criterion(logits, yb) # dY не нужен

        bs = xb.shape[0]
        total_loss += loss_val * bs
        total_correct += int((logits.argmax(axis=1) == yb.argmax(axis=1)).sum())
        total += bs
    return total_loss / total, total_correct / total
