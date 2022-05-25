import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predicted, real):
        return self.criterion(predicted.view(-1), real.view(-1))


class GeneratorLoss(nn.Module):
    def __init__(self, l1_const=100):
        super(GeneratorLoss, self).__init__()

        self.l1_const = l1_const
        self.L1Loss = nn.L1Loss()
        self.GANLoss = GANLoss()

    def forward(self, fake_labels, discriminator_predictions, generated_ab, real_ab):
        return (
                    self.GANLoss(discriminator_predictions, fake_labels) +
                    self.l1_const * self.L1Loss(generated_ab, real_ab)
               )


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.loss = GANLoss()

    def forward(self, pred_labels, real_labels):
        return self.loss(pred_labels, real_labels)