import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from helpers.evaluation import compute_accuracy, get_all_preds_and_targets
from helpers.plotting import plot_intermediate_results


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc'):
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []

    # Add lists to store precision, recall, and F1 scores for validation
    valid_precision_list, valid_recall_list, valid_f1_list = [], [], []

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)

            # Now, compute predictions and targets for validation set to calculate precision, recall, and F1
            valid_preds, valid_targets = get_all_preds_and_targets(model, valid_loader, device)

            valid_precision = precision_score(valid_targets.numpy(), valid_preds.numpy(), average='macro')
            valid_recall = recall_score(valid_targets.numpy(), valid_preds.numpy(), average='macro')
            valid_f1 = f1_score(valid_targets.numpy(), valid_preds.numpy(), average='macro')

            # Log the precision, recall, and F1 scores
            valid_precision_list.append(valid_precision)
            valid_recall_list.append(valid_recall)
            valid_f1_list.append(valid_f1)

            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

            print(
                f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | Train Acc: {train_acc:.2f}% | Validation Acc: {valid_acc:.2f}% | Precision: {valid_precision:.4f} | Recall: {valid_recall:.4f} | F1: {valid_f1:.4f}')

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

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

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list, valid_precision_list, valid_recall_list, valid_f1_list
