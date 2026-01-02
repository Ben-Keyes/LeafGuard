# General
import numpy as np
# Torch
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

# Metrics metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# For saving the class file
import json
import os

# Helps with running the 2 workers
if __name__ == '__main__':

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
     # Needs normalised aswell
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
        r"C:\Users\benke\Datasets\plantvillage\unaugmented",
        transform=None
    )

    # Save class names in the same folder as the model
    #MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    #class_names_path = os.path.join(MODEL_DIR, "class_names.json")

    #with open(class_names_path, "w") as f:
        #json.dump(dataset.classes, f)

    #print(f"Saved class names to {class_names_path}")

    # Number of samples
    num_samples = len(dataset)

    # Shuffle once for randomised training, validation and testing
    indices = np.random.permutation(num_samples)

    # Splits (70% train, 15% validation and 15% testing)
    train_split = int(0.70 * num_samples)
    val_split   = int(0.85 * num_samples)

    train_indices = indices[:train_split]
    val_indices   = indices[train_split:val_split]
    test_indices  = indices[val_split:]

    # Debugging
    print("Classes:", dataset.classes)
    print("Total:", len(dataset))
    print("Training size:", len(train_indices))
    print("Validation size:", len(val_indices))
    print("Testing size:", len(test_indices))

    # Apply transforms after splitting
    train_dataset = datasets.ImageFolder(
        r"C:\Users\benke\Datasets\plantvillage\unaugmented",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        r"C:\Users\benke\Datasets\plantvillage\unaugmented",
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        r"C:\Users\benke\Datasets\plantvillage\unaugmented",
        transform=test_transform
    )
    # A good batch size to use
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=2
    )

    # Making the model (ResNet18)
    num_classes = len(dataset.classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze all layers except the final fully connected layer
    #for param in model.parameters():
        #param.requires_grad = False
    #for param in model.fc.parameters():
        #param.requires_grad = True

    # Partial freezing, all but fc and layer4
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

    print("\n ResNet18 Initialised - now training \n")

    # Device selection (CPU cus I don't have GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Use this if fully frozen
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Use me if unfreezing fc and layer4
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )
    # Typically do 2,3 or 5
    num_epochs = 5
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Stopping training if no noticable improvement
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    best_model_path = "leaf_resnet18_best_layer4_fc.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

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
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)

                # Top-1
                confidence, predicted = torch.max(probs, 1)

                # Top3
                top3 = torch.topk(probs, k=3, dim=1).indices  # shape: [batch, 3]
                top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

                val_confidences.extend(confidence.cpu().numpy())

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_indices)
        epoch_val_acc = val_correct / val_total
        epoch_val_top3 = top3_correct / val_total

        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Performance metrics per epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"Val Top-3: {epoch_val_top3 * 100:.2f}%"
        )
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (val loss {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No val loss improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Saving the model to file
    #torch.save(model.state_dict(), "leaf_resnet18_3_epoch_layer4_fc.pth")
    #print("Model saved as leaf_resnet18_3_epoch_layer4_fc.pth")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    # Testing loop with confidence scores and metrics
    model.eval()
    test_correct = 0
    test_total = 0
    test_confidences = []

    #Storinge predictions/labels for confusion matrix + report
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
