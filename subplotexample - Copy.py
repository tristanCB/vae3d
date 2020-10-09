# Python 3
# Taken from: https://stackoverflow.com/questions/29832055/animated-subplots-using-matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
# Generates a gif version of the Variational AutoEncoder example from keras 
# using a VAE having 3 latent dimensions. this script subdivides MNIST digit data 
# sorted based on third latent dimension into equivalent n arrays (bins) and 
# overlays the latent dimensions onto a plane of sampled decoded digits. 
# When plotting the decoded digits, the mean of the third latent dimention of the nth 
# array is taken.

## User defined vars ##
# How many images to generate
bins = 120
# For the 3D scatter to rotate we set an angle grater than 1
change_in_angle = 2.5
n = 15          # The amount of subdivision used to plot the decoded space
digit_size = 28 #
scale = 4       # latent dimension range
figsize = 15    # 
figure = np.zeros((digit_size * n, digit_size * n))

# Set decoder as a plane. These are the set of x and y values that are kept constant.
# We will be sweeping in z directions (z dim 2)
grid_x = np.linspace(-scale, scale, n)
grid_y = np.linspace(-scale, scale, n)[::-1] # all elements, reversed

# Load the trained encoder and decoder model made using vaetfexample.py
encoder = tf.keras.models.load_model("ENCODER")
decoder = tf.keras.models.load_model("DECODER")

# Load the toy dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

# Get latent dims of x_train
z_mean, _, z = encoder.predict(x_train)

# Stack the labels to sort dataset 
zplusl = np.hstack((z,np.expand_dims(y_train, axis=1)))
# print("zplus =" , zplusl.shape)

# Sort acording the infered 3rd latent dimention of the digit
zplusl_sorted = zplusl[zplusl[:,2].argsort()]

# Subdivides the sorted dataset into equal parts (amount of bins)
data_bins = np.vsplit(zplusl_sorted, bins)

# Function of plot the decoded space for a fixed third dimension
def latent_space(decoder, ki):
    # display a n*n 2D manifold of digits
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi, ki]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    return figure

# %% Start Plotting
# ax1 is the figure plotting the decoded space, ax2 is the 3D latent space
fig = plt.figure(figsize = (16,7))
ax2 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122)

## Formatting axis which will displey the decoder
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)

# Small funtion to format axis
def formatAx1():
    ax1.set_xticks(pixel_range, minor=False)
    ax1.set_xticklabels(sample_range_x, fontdict=None, minor=False)
    ax1.set_yticks(pixel_range, minor=False)
    ax1.set_yticklabels(sample_range_y, fontdict=None, minor=False)
    ax1.margins(x=0, y=0)
    ax1.set_xlabel("z[0]")
    ax1.set_ylabel("z[1]")

## Formatting axis which will display the latent space
ax2.set_xlim([-scale,scale])
ax2.set_ylim([-scale,scale])
ax2.set_zlim([-scale,scale])
ax2.set_xlabel("z[0]")
ax2.set_ylabel("z[1]")
ax2.set_zlabel("z[2]")

def generate_frame(i):
    # Fixed third latent dimension based on the average values considered for a given bin
    print(i)

    data_bin_mean_z2 = np.mean(data_bins[i][:,2])
    # The fixed z dimension which is given to the decoder
    ax1.set_title(f"z[2] = {data_bin_mean_z2:.2f}")
    # Plot the average z[2] of the plotted batch
    ax1.imshow(latent_space(decoder, data_bin_mean_z2), cmap=cm.gray)
    ax1.set_autoscale_on(False)

    # Add the batch to the 3D scatter
    ax2.scatter(data_bins[i][:,0], data_bins[i][:,1],data_bins[i][:,2], alpha=0.5, c=data_bins[i][:,3], vmin=0, vmax=10) #  c=y_train
    # Rotate one frame CCW
    ax2.view_init(30, i*change_in_angle)

    ## Test for scaling
    # xs = np.asarray([0,3,-3,-3])
    # ys = np.asarray([0,3,-3,3])
    
    # The dimensions for the first and second latent dimensions for the bin we are plotting
    dim0 = data_bins[i][:,0]
    dim1 = data_bins[i][:,1]

    # Linearly transales the dim0 and 1 from the range [-scale, scale] to [pixel_range[0], pixel_range[-1]]
    # This ensures the images corespond more of less to the decoded digit show in this axis.
    # txs = pixel_range[0] + ((-dim0 + scale) * (pixel_range[-1]-pixel_range[0]))/ (scale*2)
    # tys = pixel_range[0] + ((-dim1 + scale) * (pixel_range[0]-pixel_range[-1]))/ (scale*2)

    # Plot 2D scatter on decoder
    ax1.scatter(dim0, dim1, alpha=0.25, c=data_bins[i][:,3], vmin=0, vmax=10)
    
    # Reformat axis so the labels do not change...
    formatAx1()

# Generates a frame in the animation
for i in range(bins):
    generate_frame(i)
    plt.savefig(f'./TMP/{str(i).zfill(4)}.png')
    ax1.clear()

