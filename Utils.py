"""
Author : Chinmay Dharmik
Author's Note:
        This class is a general-purpose utility for working with PyTorch models in the context of training,
        testing, and adversarial attacks. It provides methods for these tasks and can be extended or
        customized for specific use cases.
        This class is not intended to be used as-is.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt


class Utils:
    """
    Utility class for training, testing, and adversarial attacks on a PyTorch neural network model.

    Args:
        model (nn.Module): The neural network model to be used.
    """

    def __init__(self, model):
        """
        Initializes the utility class with a given neural network model.

        Args:
            model (nn.Module): The neural network model to be used.
        """
        self.model = model()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()

    def train(self, train_loader,  num_epochs=40):
        """
        Trains the neural network model.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): Number of training epochs (default is 40).

        Returns:
            nn.Module: The trained neural network model.
        """
        print("Training the Model...")
        print("-" * 30)

        train_loss_log = []
        train_accuracy_log = []
        time_per_epoch_log = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            start_time = time.time()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, predicted = output.max(1)
                total_train += target.size(0)
                correct_train += predicted.eq(target).sum().item()

            end_time = time.time()
            train_accuracy = 100.0 * correct_train / total_train
            time_per_epoch = end_time - start_time

            train_loss_log.append(train_loss / len(train_loader))
            train_accuracy_log.append(train_accuracy)
            time_per_epoch_log.append(time_per_epoch)

            print('Epoch: {}/{}...'.format(epoch, num_epochs),
                  'Loss: {:.4f}'.format(train_loss / len(train_loader)),
                  'Train Accuracy: {:.2f}%'.format(train_accuracy),
                  'Time per Epoch: {:.2f} seconds'.format(time_per_epoch), sep='\t|\t')

        return self.model

    def test(self, testloader):
        """
        Tests the neural network model on a test dataset.

        Args:
            testloader (DataLoader): DataLoader for the test dataset.
        """
        # prepare to count predictions for each class
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        print("Testing the Model...")
        print("-" * 30)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def untargeted_attack(self, images, labels, epsilon):
        """
        Performs untargeted adversarial attacks on input images.

        Args:
            images (Tensor): Input images.
            labels (Tensor): True labels for the images.
            epsilon (float): Perturbation strength.

        Returns:
            Tensor: Adversarial images.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad_(True)

        outputs = self.model(images)

        self.model.zero_grad()
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        attack_images = images + epsilon*images.grad.sign()
        attack_images = torch.clamp(attack_images, 0, 1)

        return attack_images

    def targeted_attack(self, images, target_class, epsilon):
        """
        Performs targeted adversarial attacks on input images.

        Args:
            images (Tensor): Input images.
            target_class (int): Target class for the attack.
            epsilon (float): Perturbation strength.

        Returns:
            Tensor: Adversarial images.
        """
        images = images.to(self.device)
        batch_size = images.shape[0]
        target_labels = torch.tensor(
            [target_class] * batch_size).to(self.device)
        images.requires_grad_(True)

        outputs = self.model(images)

        self.model.zero_grad()
        # Targeted loss with the specified target_label
        loss = nn.CrossEntropyLoss()(outputs, target_labels)
        loss.backward()

        attack_images = images + epsilon * images.grad.sign()
        attack_images = torch.clamp(attack_images, 0, 1)

        return attack_images

    def test_target(self, testloader, Target, epsilon=0.001):
        """
        Tests the model with targeted adversarial attacks.

        Args:
            testloader (DataLoader): DataLoader for the test dataset.
            Target (int): Target class for the attack.
            epsilon (float): Perturbation strength.
        """
        # prepare to count predictions for each class
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        print("Testing the Model...")
        print("-" * 30)
        for data in testloader:
            images, labels = data
            images = self.targeted_attack(
                images, Target, epsilon).to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def test_untargeted(self, testloader, epsilon=0.001):
        """
        Tests the model with untargeted adversarial attacks.

        Args:
            testloader (DataLoader): DataLoader for the test dataset.
            epsilon (float): Perturbation strength.
        """
        # prepare to count predictions for each class
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        print("Testing the Model...")
        print("-" * 30)
        for data in testloader:
            images, labels = data
            images = self.untargeted_attack(
                images, labels, epsilon).to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def train_adversarial(self, train_loader, epsilon, alpha=0.5, num_epochs=40):
        """
        Trains the model with adversarial examples.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            epsilon (float): Perturbation strength.
            alpha (float): Weighting factor for combining original and adversarial loss (default is 0.5).
            num_epochs (int): Number of training epochs (default is 40).

        Returns:
            nn.Module: The trained neural network model.
        """
        print("Training the Model with Adversarial Training...")
        print("-" * 50)

        train_loss_log = []
        train_accuracy_log = []
        time_per_epoch_log = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.model.to(self.device)
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            start_time = time.time()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Generate adversarial examples
                adv_data = self.untargeted_attack(data, target, epsilon)

                self.optimizer.zero_grad()

                # Forward pass with original data
                output = self.model(data)
                loss_original = self.criterion(output, target)

                # Forward pass with adversarial data
                output_adv = self.model(adv_data)
                loss_adversarial = self.criterion(output_adv, target)

                # Combine original and adversarial loss
                loss = alpha * loss_original + (1 - alpha) * loss_adversarial

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, predicted = output.max(1)
                total_train += target.size(0)
                correct_train += predicted.eq(target).sum().item()

            end_time = time.time()
            train_accuracy = 100.0 * correct_train / total_train
            time_per_epoch = end_time - start_time

            train_loss_log.append(train_loss / len(train_loader))
            train_accuracy_log.append(train_accuracy)
            time_per_epoch_log.append(time_per_epoch)

            print('Epoch: {}/{}...'.format(epoch, num_epochs),
                  'Loss: {:.4f}'.format(train_loss / len(train_loader)),
                  'Train Accuracy: {:.2f}%'.format(train_accuracy),
                  'Time per Epoch: {:.2f} seconds'.format(time_per_epoch), sep='\t|\t')

        return self.model
