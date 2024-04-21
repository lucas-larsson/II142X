import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from helpers.evaluation import compute_accuracy, get_all_preds_and_targets
from helpers.plotting import plot_intermediate_results


def train_model(model, num_epochs, train_loader, valid_loader, test_loader, optimizer,
                device, logging_interval=50, scheduler=None, scheduler_on='valid_acc'):
    start_time = time.time()

    minibatch_loss_list, train_acc_list, valid_acc_list, valid_precision_list, valid_recall_list, valid_f1_list = [], [], [], [], [], []

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            # Forward and backpropagation
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device)
            valid_acc = compute_accuracy(model, valid_loader, device)
            valid_preds, valid_targets = get_all_preds_and_targets(model, valid_loader, device)
            valid_precision = precision_score(valid_targets.cpu().numpy(), valid_preds.cpu().numpy(), average='macro')
            valid_recall = recall_score(valid_targets.cpu().numpy(), valid_preds.cpu().numpy(), average='macro')
            valid_f1 = f1_score(valid_targets.cpu().numpy(), valid_preds.cpu().numpy(), average='macro')

            valid_precision_list.append(valid_precision)
            valid_recall_list.append(valid_recall)
            valid_f1_list.append(valid_f1)

            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Validation Acc: {valid_acc:.2f}% '
                  f'| Precision: {valid_precision:.4f} '
                  f'| Recall: {valid_recall:.4f} '
                  f'| F1: {valid_f1:.4f}')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        if (epoch + 1) % 5 == 0:
            plot_intermediate_results(minibatch_loss_list, train_acc_list, valid_acc_list, epoch + 1)

        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    # Evaluate on test set
    model.eval()
    test_acc = compute_accuracy(model, test_loader, device)
    test_preds, test_targets = get_all_preds_and_targets(model, test_loader, device)
    test_precision = precision_score(test_targets.cpu().numpy(), test_preds.cpu().numpy(), average='macro')
    test_recall = recall_score(test_targets.cpu().numpy(), test_preds.cpu().numpy(), average='macro')
    test_f1 = f1_score(test_targets.cpu().numpy(), test_preds.cpu().numpy(), average='macro')

    print(
        f'Test Metrics | Accuracy: {test_acc:.2f}% | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}')

    return minibatch_loss_list, train_acc_list, valid_acc_list, valid_precision_list, valid_recall_list, valid_f1_list, test_acc, test_precision, test_recall, test_f1
