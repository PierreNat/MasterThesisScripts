
import numpy as np
import torch
from numpy.random import uniform
import neural_renderer as nr
import matplotlib.pyplot as plt
import tqdm
import math
import matplotlib.image as mpimg
import random
from scipy.misc import imsave
import imageio
import os
import glob
import argparse
from skimage.io import imread, imsave
from utils_functions.camera_settings import camera_setttings

pi = math.pi
current_dir = os.path.dirname(os.path.realpath(__file__))

result_dir = os.path.join(current_dir, 'Npydatabase')

def PointsInCircum(r,n=1000):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

def main():

    cubes_database = []
    sils_database = []
    params_database = []
    im_nr = 1

    vertices_1, faces_1, textures_1 = nr.load_obj("3D_objects/wrist.obj", load_texture=True) #, texture_size=4)
    print(vertices_1.shape)
    print(faces_1.shape)
    vertices_1 = vertices_1[None, :, :]  # add dimension
    faces_1 = faces_1[None, :, :]  #add dimension
    textures_1 = textures_1[None, :, :]  #add dimension
    nb_vertices = vertices_1.shape[0]

    print(vertices_1.shape)
    print(faces_1.shape)


    nb_im = 1000

    file_name_extension = 'exampleOfNameForTheTimelapse'.format(nb_im)
    parser = argparse.ArgumentParser()
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(result_dir, 'Animation_{}.gif'.format(file_name_extension)))
    args = parser.parse_args()

    #init and create renderer object
    R = np.array([np.radians(0), np.radians(0), np.radians(0)])  # angle in degree
    t = np.array([0, 0, 0])  # translation in meter
    cam = camera_setttings(R=R, t=t, vert=nb_vertices)
    renderer = nr.Renderer(image_size=512, camera_mode='projection', dist_coeffs=None,
                           K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=1, background_color=[0, 0, 0], #background is filled now with  value 0-1 instead of 0-255
                           # changed from 0-255 to 0-1
                           far=1000, orig_size=512,
                           light_intensity_ambient=1.0, light_intensity_directional=0, light_direction=[0, 1, 0],
                           light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])

    prevA = 0
    prevB = 0
    prevG = 0
    circlepoint = PointsInCircum(1.4, nb_im)

    Z_sinus = np.linspace(0, 360, nb_im)
    Z_sinus = 6 + np.sin(np.radians(Z_sinus))
    circlepoint_alpha = np.linspace(0, 180, nb_im)
    circlepoint_beta = np.linspace(180, 0, nb_im)
    circlepoint_gamma = np.linspace(0, 180, nb_im)
    print('z axis translation min max are {} and {}'.format(Z_sinus.min(), Z_sinus.max())) #5.000000015261379 and 6.999998763828597

    loop = tqdm.tqdm(range(0, nb_im))

    for i in loop:
        # define transfomration parameter randomly uniform

#------INSERT HERE SPECIFIC VALUE OF TRANSLATION AND ROTATION -----------------------------------------------

        alpha =0# circlepoint_alpha[i]
        beta = 0 #circlepoint_beta[i]
        gamma = circlepoint_gamma[i]

        x = circlepoint[i][0]
        y = circlepoint[i][1]
        z = Z_sinus[i] #1000t was done with value between 7 and 10, Rot and trans between 5 10


        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([x, y, z])  # translation in meter

        Rt = np.concatenate((R, t), axis=None).astype(np.float16)  # create one array of parameter in radian, this arraz will be saved in .npy file

        cam = camera_setttings(R=R, t=t, vert=nb_vertices) # degree angle will be converted  and stored in radian

        images_1 = renderer(vertices_1, faces_1, textures_1,
                            K=torch.cuda.FloatTensor(cam.K_vertices),
                            R=torch.cuda.FloatTensor(cam.R_vertices),
                            t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]


        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0-1
        img = image
        image = (image*255).astype(np.uint8) #cast from float32 255.0 to 255 uint8

        sils_1 = renderer(vertices_1, faces_1, textures_1,
                          mode='silhouettes',
                          K=torch.cuda.FloatTensor(cam.K_vertices),
                          R=torch.cuda.FloatTensor(cam.R_vertices),
                          t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        sil = sils_1.detach().cpu().numpy().transpose((1, 2, 0))
        sil = np.squeeze((sil * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]

        #grow the list of cube, silhouette and parameters
        cubes_database.extend(image)
        sils_database.extend(sil)
        params_database.extend(Rt)

        im_nr = im_nr+1

        # #put surgical iamge as a background
        # BinarySil3layermask = (np.array([sil, sil,sil])).transpose((1, 2, 0))/255
        # # plt.imshow(BinarySil3layermask)
        # numberbackground = random.randint(1,8)
        # backgroundImg = mpimg.imread("3D_objects/background{}.jpg".format(numberbackground))
        #
        # sx = backgroundImg.shape[0]
        # sy = backgroundImg.shape[1]
        #
        # moveX = random.randint(0,sx-512)
        # moveY = random.randint(0,sy-512)
        # # print(moveX, moveY)
        # cropedbackgroundImg = backgroundImg[moveX:moveX+512, moveY:moveY+512, :]
        # maskedbackground = np.multiply((BinarySil3layermask *-1+1), cropedbackgroundImg/255)
        # imWithBackground = (image + (maskedbackground*255)).astype(np.uint8)
        # # plt.imshow(imWithBackground)
        # image = imWithBackground


        # cubes_database.extend(image)

        # if(im_nr%50 == 0):
        #     fig = plt.figure()
        #     fig.add_subplot(2, 1, 1)
        #     plt.imshow(image)
        #     imageio.imwrite("3D_objects/{}_ref.png".format(file_name_extension), image)
        #
        #     fig.add_subplot(2, 1, 2)
        #     plt.imshow(sil, cmap='gray')
        #     plt.show()
        #     plt.close(fig)
        #
        imsave('/tmp/_tmp_%04d.png' % i, img)


# save database
    make_gif(args.filename_output)

# reshape in the form (nbr of image, x dim, y dim, layers)
    cubes_database = np.reshape(cubes_database, (im_nr-1, 512, 512, 3)) # 3 channel rgb
    sils_database = np.reshape(sils_database, (im_nr-1, 512, 512)) #binary mask monochannel
    params_database = np.reshape(params_database,(im_nr-1, 6)) #array of 6 params
    np.save('Npydatabase/cubes_{}.npy'.format(file_name_extension), cubes_database)
    np.save('Npydatabase/sils_{}.npy'.format(file_name_extension), sils_database)
    np.save('Npydatabase/params_{}.npy'.format(file_name_extension), params_database)
    print('images saved')


if __name__ == '__main__':
    main()
