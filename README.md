# dataset 
I took 2000 images from the Stanford dogs dataset, and synthetically added noise to it.

noise types used are: (impuls noise - gaussian noise - uniform noise - rayleigh noise - erling noise - exponential noise).

images shape: 3x128x128.

number of images: 2000 * 6 = 12000.

I normalized images for better convergence. 

#challenge 
I insisted on working on this project locally on my device (16 GB of RAM) to face the challenge of optimization. Not claiming I didn't have any system crashes in the middle, but from mistakes we learn :) 

I used multiprocessing to create our 12000 RGB 128x128 images (without multiprocessing it took about 30m, but with multiporcessing it took only around 5m).

# network 
## archeticture
I used U-net architecture, Mixing CNN and auto-encoder in one architecture.

U-net example:
<img width="1494" height="748" alt="image" src="https://github.com/user-attachments/assets/8b14676b-8ac3-4bba-b09f-e9892756471f" />


The idea is simple, use CNNs to increase the number of filters and decrease filter size, untill you hit a bottleneck (the encoding part).

after that, user Convolution transpose to invert the first part (the decoder part).

I used batch-norm for stable training and skip connections to prevent vanshing gradient. 



## Loss 
I used L1 to compute Loss. 

# optimizer
Adam optimizer was used.



# results 

<img width="1161" height="793" alt="image" src="https://github.com/user-attachments/assets/9000a2e4-fbc9-4e9a-9cd9-2c09176a8b47" />

<img width="1157" height="799" alt="image" src="https://github.com/user-attachments/assets/02a3d7f7-ec69-4656-a77b-6c6ba648bdaa" />

<img width="1163" height="798" alt="image" src="https://github.com/user-attachments/assets/6aa54d58-0fca-40e1-840e-361d91f8eeb8" />

<img width="1165" height="801" alt="image" src="https://github.com/user-attachments/assets/d327c34f-10af-4577-b21f-6286c9bd1c10" />

# effect on clear images
