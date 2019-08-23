
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import  pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils_functions.R2Rmat import R2Rmat
from numpy.random import uniform
import matplotlib2tikz

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def RolAv(list, window = 2):

    mylist = list
    # print(mylist)
    N = window
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return moving_aves

def train_renderV2(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im):
    # monitor loss functions as the training progresses

    loop = n_epochs
    Step_Val_losses = []
    current_step_loss = []
    current_step_Test_loss = []
    Test_losses = []
    Epoch_Val_losses = []
    Epoch_Test_losses = []
    count = 0
    testcount = 0
    Im2ShowGT = []
    Im2ShowGCP = []
    LastEpochTestCPparam = []
    LastEpochTestGTparam = []
    numbOfImageDataset = number_train_im
    renderCount = 0
    regressionCount = 0
    renderbar = []
    regressionbar = []
    lr= 0.0001 #best at 0.0001 with translation only with span model.t[2] > 4 and model.t[2] < 8 and torch.abs(model.t[0]) < 2 and torch.abs(model.t[1]) < 2):

    output_dir_model = 'models/render/{}'.format(fileExtension)
    mkdir_p(output_dir_model)
    output_dir_results = 'results/render/{}'.format(fileExtension)
    mkdir_p(output_dir_results)


    for epoch in range(n_epochs):

        ## Training phase
        model.train()
        print('train phase epoch {}/{}'.format(epoch, n_epochs))

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        for image, silhouette, parameter in t:
            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]
            numbOfImage = image.size()[0]
            # loss = nn.MSELoss()(params, parameter).to(device)

            for i in range(0,numbOfImage):
                #create and store silhouette
                model.t = params[i, 3:6]
                R = params[i, 0:3]
                model.R = R2Rmat(R)  # angle from resnet are in radian

                current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t, mode='silhouettes').squeeze()
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                if (model.t[2] > 4 and model.t[2] < 8 and torch.abs(model.t[0]) < 2 and torch.abs(model.t[1]) < 2):
                # if (epoch > 0):
                #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    optimizer.zero_grad()
                    if (i == 0):
                        loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
                    else:
                        loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device)
                    renderCount += 1
                else:
                    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    optimizer.zero_grad()
                    if (i == 0):
                        loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                        # loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                    else:
                        loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                        # loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
                    regressionCount += 1

            loss.backward()
            optimizer.step()

            Step_Val_losses.append(loss.detach().cpu().numpy())  # contain all step value for all epoch
            current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

#        if (epoch % 5 == 0 and epoch > 2):
#            if (lr > 0.000001):
#                lr = lr / 10
#            print('update lr, is now {}'.format(lr))

        epochValloss = np.mean(current_step_loss)
        current_step_loss = []
        Epoch_Val_losses.append(epochValloss)  # most significant value to store
        # print(epochValloss)

        print(Epoch_Val_losses)
        # print(renderCount, regressionCount)

        renderbar.append(renderCount)
        regressionbar.append(regressionCount)
        renderCount = 0
        regressionCount = 0

        torch.save(model.state_dict(),
                   '{}/{}_{}epoch_{}_Temp_train_{}_{}batchs_{}epochs_Render.pth'.format(output_dir_model, fileExtension, epoch, date4File,
                                                                                                      cubeSetName,
                                                                                                      str(batch_size),
                                                                                                      str(n_epochs),
                                                                                                      ))
        # validation phase
        print('test phase epoch epoch {}/{}'.format(epoch, n_epochs))
        model.eval()

        t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
        for image, silhouette, parameter in t:

            Test_Step_loss = []
            numbOfImage = image.size()[0]

            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]
            # print(np.shape(params))

            for i in range(0,numbOfImage):
                #create and store silhouette
                model.t = params[i, 3:6]
                R = params[i, 0:3]
                model.R = R2Rmat(R)  # angle from resnet are in radian

                current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t, mode='silhouettes').squeeze()
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                if (i == 0):
                    loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
                else:
                    loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device)

            Test_Step_loss.append(loss.detach().cpu().numpy())

            if (epoch == n_epochs - 1):  # if we are at the last epoch, save param to plot result

                LastEpochTestCPparam.extend(params.detach().cpu().numpy())
                LastEpochTestGTparam.extend(parameter.detach().cpu().numpy())

            Test_losses.append(loss.detach().cpu().numpy())
            current_step_Test_loss.append(loss.detach().cpu().numpy())
            testcount = testcount + 1

        epochTestloss = np.mean(current_step_Test_loss)
        current_step_Test_loss = []
        Epoch_Test_losses.append(epochTestloss)  # most significant value to store

# ----------- plot some result from the last epoch computation ------------------------

        # print(np.shape(LastEpochTestCPparam)[0])
    nim = 4
    for i in range(0, nim):
        print('saving image to show')
        pickim = int(uniform(0, np.shape(LastEpochTestCPparam)[0] - 1))
        # print(pickim)

        model.t = torch.from_numpy(LastEpochTestCPparam[pickim][3:6]).to(device)
        R = torch.from_numpy(LastEpochTestCPparam[pickim][0:3]).to(device)
        model.R = R2Rmat(R)  # angle from resnet are in radia
        imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)

        model.t = torch.from_numpy(LastEpochTestGTparam[pickim][3:6]).to(device)
        R = torch.from_numpy(LastEpochTestGTparam[pickim][0:3]).to(device)
        model.R = R2Rmat(R)  # angle from resnet are in radia
        imgGT, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)

        imgCP = imgCP.squeeze()  # float32 from 0-1
        imgCP = imgCP.detach().cpu().numpy().transpose((1, 2, 0))
        imgCP = (imgCP * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
        imgGT = imgGT.squeeze()  # float32 from 0-1
        imgGT = imgGT.detach().cpu().numpy().transpose((1, 2, 0))
        imgGT = (imgGT * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
        Im2ShowGT.append(imgCP)
        Im2ShowGCP.append(imgGT)

        a = plt.subplot(2, nim, i + 1)
        plt.imshow(imgGT)
        a.set_title('GT {}'.format(i))
        plt.xticks([0, 512])
        plt.yticks([])
        a = plt.subplot(2, nim, i + 1 + nim)
        plt.imshow(imgCP)
        a.set_title('Rdr {}'.format(i))
        plt.xticks([0, 512])
        plt.yticks([])

    plt.savefig('{}/image_render_{}batch_{}epochs_{}.png'.format(output_dir_results,batch_size, n_epochs, fileExtension), bbox_inches = 'tight', pad_inches = 0.05)

#-----------plot and save section ------------------------------------------------------------------------------------

    fig, (p1, p2, p3, p4) = plt.subplots(4, figsize=(15, 10))  # largeur hauteur
    fig.suptitle("Render, training {} epochs with {} images, lr={} ".format(n_epochs, numbOfImageDataset, lr), fontsize=14)

    moving_aves = RolAv(Step_Val_losses, window=50)
    ind = np.arange(n_epochs)  # index

    p1.plot(np.arange(np.shape(moving_aves)[0]), moving_aves, label="step Loss rolling average")
    p1.set(ylabel='BCE Step Loss')
    p1.set_yscale('log')
    p1.set(xlabel='Steps')
    p1.set_ylim([0, 20])
    p1.legend()  # Place a legend to the right of this smaller subplot.

    # subplot 2
    p2.plot(np.arange(n_epochs), Epoch_Val_losses, label="Render epoch Loss")
    p2.set(ylabel=' Mean of BCE training step loss')
    p2.set(xlabel='Epochs')
    p2.set_ylim([0, 20])
    p2.set_xticks(ind)
    p2.legend()

    # subplot 3

    width = 0.35
    p3.bar(ind, renderbar, width, color='#d62728', label="render")
    height_cumulative = renderbar
    p3.bar(ind, regressionbar, width, bottom=height_cumulative, label="regression")
    p3.set(ylabel='render/regression call')
    p3.set(xlabel='Epochs')
    p3.set_ylim([0, numbOfImageDataset])
    p3.set_xticks(ind)
    p3.legend()

    # subplot 4
    p4.plot(np.arange(n_epochs), Epoch_Test_losses, label="Render Test Loss")
    p4.set(ylabel='Mean of BCE test step loss')
    p4.set(xlabel='Epochs')
    p4.set_ylim([0, 10])
    p4.legend()


    plt.show()

    fig.savefig('{}/render_{}batch_{}epochs_{}.png'.format(output_dir_results, batch_size, n_epochs, fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
    matplotlib2tikz.save("{}/render_{}batch_{}epochs_{}.tex".format(output_dir_results, batch_size, n_epochs, fileExtension),figureheight='5.5cm', figurewidth='15cm')


    torch.save(model.state_dict(),
               '{}/{}_{}_FinalModel_train_{}_{}batchs_{}epochs_Render.pth'.format(output_dir_model, fileExtension, date4File,
                                                                                        cubeSetName,
                                                                                        str(batch_size),
                                                                                        str(n_epochs),
                                                                                        ))
    print('parameters saved')
