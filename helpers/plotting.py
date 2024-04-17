import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_metric(metrics_vgg16, metrics_vgg23, metric_name, results_dir):
    """
    Plot given metric for both VGG-16 and VGG-23 models on the same graph.
    """
    epochs = np.arange(1, len(metrics_vgg16) + 1)
    plt.figure()
    plt.plot(epochs, metrics_vgg16, label=f'{metric_name} VGG-16', marker='o', color='blue')
    plt.plot(epochs, metrics_vgg23, label=f'{metric_name} VGG-23', marker='x', color='green')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, f'comparison_{metric_name.lower()}.pdf'))
    plt.show()
    plt.close()


def plot_accuracy(train_acc_vgg16, valid_acc_vgg16, train_acc_vgg23, valid_acc_vgg23, results_dir):
    """
    Plot training and validation accuracy for both models.
    """
    epochs = np.arange(1, len(train_acc_vgg16) + 1)
    plt.figure()
    plt.plot(epochs, train_acc_vgg16, 'b-', label='Training Accuracy VGG-16')
    plt.plot(epochs, valid_acc_vgg16, 'b--', label='Validation Accuracy VGG-16')
    plt.plot(epochs, train_acc_vgg23, 'g-', label='Training Accuracy VGG-23')
    plt.plot(epochs, valid_acc_vgg23, 'g--', label='Validation Accuracy VGG-23')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison over Epochs')
    plt.legend()
    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, 'comparison_accuracy.pdf'))
    plt.show()
    plt.close()


def plot_training_loss(minibatch_loss_list_vgg16, minibatch_loss_list_vgg23, num_epochs, iter_per_epoch,
                       results_dir=None, averaging_iterations=100):
    """
    Plot training loss for both VGG-16 and VGG-23 models on the same graph.
    """
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    # Plot loss for VGG-16
    ax1.plot(range(len(minibatch_loss_list_vgg16)),
             minibatch_loss_list_vgg16, label='Minibatch Loss VGG-16', color='blue')

    # Plot loss for VGG-23
    ax1.plot(range(len(minibatch_loss_list_vgg23)),
             minibatch_loss_list_vgg23, label='Minibatch Loss VGG-23', color='green')

    # Running average for VGG-16
    ax1.plot(np.convolve(minibatch_loss_list_vgg16,
                         np.ones(averaging_iterations) / averaging_iterations,
                         mode='valid'),
             label='Running Average VGG-16', color='lightblue')

    # Running average for VGG-23
    ax1.plot(np.convolve(minibatch_loss_list_vgg23,
                         np.ones(averaging_iterations) / averaging_iterations,
                         mode='valid'),
             label='Running Average VGG-23', color='lightgreen')

    ax1.legend()
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ###################
    # Set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss_comparison.pdf')
        plt.savefig(image_path)
    plt.show()
    plt.close()


def show_examples(model, data_loader, unnormalizer=None, class_dict=None):
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplots(nrows=3, ncols=5,
                             sharex=True, sharey=True)

    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False

    else:

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None):
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_intermediate_results(minibatch_loss_list, train_acc_list, valid_acc_list, epoch):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(minibatch_loss_list, label='Minibatch loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Training Accuracy', color='r')
    plt.plot(valid_acc_list, label='Validation Accuracy', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy at Epoch: {epoch}')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
