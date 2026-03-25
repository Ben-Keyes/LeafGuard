# General
import numpy as np
# Torch
import torch
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import random
# Metrics metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# For saving the class file
import json
import os

from collections import Counter

from torchvision.models import ResNet18_Weights, resnet18

TRAIN_DIR = r"C:\Users\benke\Datasets\plantvillage\unaugmented"
CUSTOM_VAL_DIR = r"C:\Users\benke\Datasets\custom"

# Helps with running the 2 workers
if __name__ == '__main__':

    # Transformations
    train_transform = transforms.Compose([
        # transforms.Resize((144, 144)),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),

        # Translation + Shear
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            shear=(-5, 5)
        ),

        transforms.ColorJitter(brightness=0.20,
                               contrast=0.20,
                               saturation=0.25,
                               hue=0.04),

        # Gamma correction
        #RandomGamma(gamma_range=(0.85, 1.15), p=0.3),

        # Photo-realistic transformations
        #transforms.RandomPerspective(distortion_scale=0.15, p=0.15),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.10),

        transforms.ToTensor(),

        # Occlusion AFTER ToTensor
        #transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), ratio=(0.5, 2.0), value='random'),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Needs normalised aswell
    test_transform = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    # Importing the dataset (unaugmented)
    dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=None
    )

    # Save class names in the same folder as the model
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(dataset.classes, f, indent=2)

    # print(f"Saved class names to {class_names_path}")

    # Number of samples
    num_samples = len(dataset)

    # Seeding split incase needed to reproduce
    np.random.seed(42)
    torch.manual_seed(42)

    # Shuffle once for randomised training, validation and testing
    indices = np.random.permutation(num_samples)

    # Splits (70% train, 30% testing and custom validation)
    train_split = int(0.70 * num_samples)
    # val_split   = int(0.85 * num_samples) -> Replaced

    train_indices = indices[:train_split]
    # val_indices   = indices[train_split:val_split] -> Replaced
    test_indices = indices[train_split:]

    # Debugging
    print("Classes:", dataset.classes)
    print("Total:", len(dataset))
    print("Training size:", len(train_indices))
    # print("Validation size:", len(val_indices)) -> Replaced (its 137 btw)
    print("Testing size:", len(test_indices))

    # Apply transforms after splitting
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        CUSTOM_VAL_DIR,
        transform=test_transform
    )

    # Forcing validation dataset to use same mapping as training
    val_dataset.class_to_idx = train_dataset.class_to_idx
    val_dataset.classes = train_dataset.classes

    val_dataset.samples = [
        (path, train_dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in val_dataset.samples
    ]
    val_dataset.targets = [y for _, y in val_dataset.samples]

    print("Val true label distribution (top 10):")
    true_counts = Counter(val_dataset.targets)
    print([(dataset.classes[i], c) for i, c in true_counts.most_common(10)])

    print("Val classes:", len(val_dataset.classes))
    print("Val samples:", len(val_dataset))

    print("Train == Val class order:", train_dataset.classes == val_dataset.classes)

    test_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=test_transform
    )

    # Highest performing batch size
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=4,
        persistent_workers=True
    )

    # Making the model (ResNet18)
    num_classes = len(dataset.classes)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer4 & fc
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

    print("\n ResNet18 Initialised - now training \n")

    # Device selection (Using GPU when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    # Class weighted loss
    # dataset.targets is a list of class indices for each sample in ImageFolder
    train_targets = np.array(dataset.targets)[train_indices]

    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)

    # Inverse square root frequency
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)

    # Normalising so average weight ~ 1
    class_weights = class_weights * (num_classes / class_weights.sum())

    # Move weights to torch tensor on device
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Clamping background class weight as it was a magnet for a lot of predictions
    bg_idx = dataset.class_to_idx["Background_without_leaves"]
    print("Background idx:", bg_idx, "old weight:", float(class_weights[bg_idx]))

    class_weights[bg_idx] = 0.2  # Keep around 0-2 to 0.5
    print("Background new weight:", float(class_weights[bg_idx]))

    print("Train class counts:", class_counts.astype(int))
    print("Class weights (first 10):", class_weights[:10].cpu().numpy())

    # Sanitty checking
    topk = torch.topk(class_weights.cpu(), k=5)
    print("Most upweighted classes:")
    for w, idx in zip(topk.values, topk.indices):
        print(dataset.classes[idx], float(w))

    # criterion = nn.CrossEntropyLoss()

    # Now using class weighted CrossEntropyLoss
    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    criterion_eval = nn.CrossEntropyLoss()  # unweighted

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Use this if fully frozen
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Use me if unfreezing fc and layer4
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 5e-4},
        {"params": model.layer4.parameters(), "lr": 3e-5},
    ], weight_decay=5e-4)

    # Trying to prevent drift after 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, threshold=1e-3
    )

    # Typically do 2,3 or 5
    num_epochs = 6
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Stopping training if no noticable improvement
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    best_model_path = "leaf_resnet18_GPU.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        val_pred_counts = Counter()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_train(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)

            if (total // batch_size) % 100 == 0:
                print(f"Processed {total}/{len(train_indices)} samples...")

            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_indices)
        epoch_train_acc = correct / total

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_confidences = []

        top3_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion_eval(outputs, labels)

                val_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)

                # Top-1
                confidence, predicted = torch.max(probs, 1)

                # Checking most predicted classes
                val_pred_counts.update(predicted.cpu().numpy().tolist())

                # Top3
                top3 = torch.topk(probs, k=3, dim=1).indices  # shape: [batch, 3]
                top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

                val_confidences.extend(confidence.cpu().numpy())

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        epoch_val_top3 = top3_correct / val_total

        scheduler.step(epoch_val_loss)

        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Performance metrics per epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"Val Top-3: {epoch_val_top3 * 100:.2f}%"
        )
        # Metrics to ensure class isn't offsetting
        print("Top val preds:", [(dataset.classes[i], c) for i, c in val_pred_counts.most_common(5)])

        # Early stopping
        min_delta = 0.005 # Prevents noise from implying a better model
        if epoch_val_acc > best_val_acc + min_delta:
            best_val_acc = epoch_val_acc
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (val accuracy {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"No val accuracy improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    # Testing loop with confidence scores and metrics
    model.eval()
    test_correct = 0
    test_total = 0
    test_confidences = []

    # Storinge predictions/labels for confusion matrix + report
    all_test_labels = []
    all_test_preds = []
    all_test_conf = []

    top3_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Top-1
            confidence, predicted = torch.max(probs, 1)

            # Top-3
            top3 = torch.topk(probs, k=3, dim=1).indices  # shape: [batch, 3]
            top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

            conf_np = confidence.cpu().numpy()
            pred_np = predicted.cpu().numpy()
            lab_np = labels.cpu().numpy()

            test_confidences.extend(conf_np)

            all_test_conf.extend(conf_np.tolist())
            all_test_preds.extend(pred_np.tolist())
            all_test_labels.extend(lab_np.tolist())

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Average Test Confidence: {np.mean(test_confidences):.3f}")

    top3_acc = 100 * top3_correct / test_total
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")

    # Confusion matrix for analysis
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)

    # Mini classification report (precision/recall/F1 per class)
    print("\nClassification Report:\n")
    print(classification_report(
        all_test_labels,
        all_test_preds,
        target_names=dataset.classes,
        digits=4
    ))

    # macro/weighted F1 (apparently helpful for many classes)
    macro_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    weighted_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # confidence analysis (correct vs incorrect)
    all_test_labels_np = np.array(all_test_labels)
    all_test_preds_np = np.array(all_test_preds)
    all_test_conf_np = np.array(all_test_conf)

    correct_mask = (all_test_labels_np == all_test_preds_np)
    if correct_mask.any():
        print(f"Avg confidence (correct):   {all_test_conf_np[correct_mask].mean():.3f}")
    if (~correct_mask).any():
        print(f"Avg confidence (incorrect): {all_test_conf_np[~correct_mask].mean():.3f}")

    # top confusions (most frequent wrong pairs)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    flat_idx = np.argsort(cm_no_diag.ravel())[::-1]
    print("\nTop 10 confusions:")
    shown = 0
    for idx in flat_idx:
        count = cm_no_diag.ravel()[idx]
        if count == 0 or shown >= 10:
            break
        true_i, pred_j = np.unravel_index(idx, cm_no_diag.shape)
        print(f"{dataset.classes[true_i]} -> {dataset.classes[pred_j]} : {count}")
        shown += 1
