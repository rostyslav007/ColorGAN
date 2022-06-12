import torch.nn as nn
import torch


def get_gradient(crit, real, fake, x, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(x, mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


def get_gen_loss(crit_fake_pred, fake, y):
    gen_loss = -torch.mean(crit_fake_pred) + nn.L1Loss()(fake, y)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda*gp

    return crit_loss


#def get_noise(batch_size, z_dim, device):
#    noise = torch.normal(0, 1, (batch_size, z_dim), device=device)
#
#    return noise

#for e in range(num_epochs):
#    for i, (x, y) in enumerate(data_loader):
#        x, y = x.cuda(), y.cuda()
#        fake_images = generator(x)

#        d_loss = 0
#        for s in range(num_critic_steps):
#            disc_real_preds = discriminator(x, y)
#            disc_fake_preds = discriminator(x, fake_images.detach())
#            epsilon = torch.rand(size=(y.shape[0], 1, 1, 1), requires_grad=True, device='cuda')
#            gradient = get_gradient(discriminator, y, fake_images.detach(), x, epsilon)
#            gp = gradient_penalty(gradient)

 #           disc_loss = get_crit_loss(disc_fake_preds, disc_real_preds, gp, gp_lambda)

 #           disc_optim.zero_grad()
 #           disc_loss.backward()
 #           disc_optim.step()
            #d_scaler.scale(disc_loss).backward()
            #d_scaler.step(disc_optim)
            #d_scaler.update()

 #           d_loss += disc_loss.item() / num_critic_steps

#        fake_images = generator(x)
#        fake_preds = discriminator(x, fake_images.detach())

#        gen_loss = get_gen_loss(fake_preds, fake_images, y)

#        gen_optim.zero_grad()
#        gen_loss.backward()
#        gen_optim.step()
        #g_scaler.scale(gen_loss).backward()
        #g_scaler.step(gen_optim)
        #g_scaler.update()

 #       disc_loss_list.append(d_loss)
 #       gen_loss_list.append(gen_loss.item())

 #       print('\ndisc loss: ', disc_loss_list[-1], ' gen loss: ', gen_loss_list[-1])
#        if e % 1 == 0 and i == 0:
#            imgs = fake_images
#            grid = make_grid((imgs * 0.5 + 0.5).cpu().detach())
#            img = torchvision.transforms.ToPILImage()(grid)
#            img.show()
