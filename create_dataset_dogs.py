import os
import numpy as np
from PIL import Image
from noise_adder import noise
import pickle
from multiprocessing import Pool
import resource
import concurrent.futures


# for i, file in enumerate(os.listdir('Images')):
#     if i == 5:
#         break
#     img = Image.open(os.path.join(os.getcwd(),'Images', file))
#     img = img.convert('RGB')
#     arr = np.array(img)
#     print(img.size)
#     print(arr.shape)

# Limit memory usage to prevent system crash (1GB per process)

# GB = 1024 * 1024 * 1024
# resource.setrlimit(resource.RLIMIT_AS, (int(1.5 * GB), int(1.5 * GB)))

#open first n images and create noisy versions for each noise type from the 6 we have
taken_images_n = 2000
data = np.zeros(shape=(taken_images_n, 3, 128, 128), dtype=np.uint8)
for i, file in enumerate(os.listdir('Images')):
    if i == taken_images_n:
        break
    im = Image.open(os.path.join(os.getcwd(), 'Images', file))
    im = im.convert('RGB')
    im = im.resize((128, 128))
    im_numpy = np.array(im)
    #reshape it to be good with tensor
    im_reshaped = np.transpose(im_numpy, (2, 0, 1)) #from (128, 128, 3) to (3, 128, 128)
    data[i] = im_reshaped


#generate noise
def get_noisy_section(noise_type: str):
    noisy_data = np.ndarray(shape=(0, 3, 128, 128))
    shape=(3, 128, 128)
    global data
    for image in data:
        if noise_type == 'G':
            noisy_image = noise('G', image, u=0.0, s=7, shape=shape)
        elif noise_type == 'I':
            noisy_image = noise('I', image, ratio=0.08, shape=shape )
        elif noise_type == 'R':
           noisy_image = noise('R', image, mode=20, shape=shape)
        elif noise_type == 'Er':
            noisy_image = noise('Er', image, k=1, theta=0.07, shape=shape)
        elif noise_type == 'Ex':
            noisy_image = noise('Ex', image, u=0.15, shape=shape)
        elif noise_type == 'U':
            noisy_image = noise('U', image, low=-0.15, high=0.15, shape=shape)
        else:
            print('wrong noise type')
            return

        noisy_data = np.vstack((noisy_data, np.reshape(noisy_image, shape=(1, 3, 128, 128))))

    return noisy_data


#create noise for the 6 noise classes:
def gaussian():
    gaussian_noisy_images = get_noisy_section('G')
    with open('gaussian.pkl', 'wb') as f:
        pickle.dump(gaussian_noisy_images, f)
    del gaussian_noisy_images #free memory space
    print('gaussian done')

def impulse():
    impulse_noisy_images = get_noisy_section('I')
    with open('impulse.pkl', 'wb') as f:
        pickle.dump(impulse_noisy_images, f)
    del impulse_noisy_images
    print('impulse done')

def rayleigh():
    rayleigh_noisy_images = get_noisy_section('R')
    with open('rayleigh.pkl', 'wb') as f:
        pickle.dump(rayleigh_noisy_images, f)
    del rayleigh_noisy_images
    print('rayleigh done')

def erling():
    Erling_noisy_images = get_noisy_section("Er")
    with open('erling.pkl', 'wb') as f:
        pickle.dump(Erling_noisy_images, f)
    del Erling_noisy_images
    print('erling done')

def expo():
    Exponential_noisy_images = get_noisy_section('Ex')
    with open('expo.pkl', 'wb') as f:
        pickle.dump(Exponential_noisy_images, f)
    del Exponential_noisy_images
    print('expo done')

def uniform():
    Uniform_noisy_images = get_noisy_section('U')
    with open('uniform.pkl', 'wb') as f:
        pickle.dump(Uniform_noisy_images, f)
    del Uniform_noisy_images
    print('uniform done')

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.submit(gaussian)
    executor.submit(impulse)
    executor.submit(rayleigh)
    executor.submit(erling)
    executor.submit(expo)
    executor.submit(uniform)
