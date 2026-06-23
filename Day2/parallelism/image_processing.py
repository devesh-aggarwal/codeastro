
import time
import os
import multiprocessing as mp
import numpy as np
import astropy.io.fits as fits
import scipy.ndimage as ndimage


####### Generate fake data ########

def generate_fake_data(filename, dims):
    """
    Generates a fake dataframe with random numbers

    Args:
        filename (str): file to save the data to
        dims (tuple): (Ny, Nx) pair that species the size of the y and x dimensions
    """
    # some complicated random image generation. Feel free to ignore.
    # coordinate system in fourier spae
    u,v = np.meshgrid(np.fft.fftfreq(dims[1]), np.fft.fftfreq(dims[0]))
    phases = np.random.uniform(0, 2*np.pi, u.shape)
    rho = np.sqrt((u*dims[1])**2 + (v*dims[0])**2)
    # suppress high frequency by a squared exponential
    spectrum = np.exp(-rho**2/(np.max(rho)/50)**2)  * np.exp(1j * phases)
    filtered = np.real(np.fft.ifft2(spectrum))

    fits.writeto(filename, filtered, overwrite=True)


# generate fake data (can choose to save it somethere else if you want)
output_folder = os.path.join(".", "fake_data")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
fileformat = os.path.join(output_folder, "fake_{0}x{1}_{2}.fits")

ny = 1000
nx = 1000
for i in range(25):
    filename = fileformat.format(ny, nx, i)
    generate_fake_data(filename, (ny, nx) )


###### Process the data #######

def process_image(frame, filtersize=50):
    """
    Run a high-pass filter on the data.
    Remove the low spatial frequency (i.e., smooth features) in the image

    Args:
        frame (np.array): a 2-D image to be processed
        fitersize (int): the size of the filter. Features smaller than the filtersize will be preserved

    Returns:
        processed_frame (np.array): a 2-D image after processing
    """
    # run a median filter to smooth the image
    frame_smooth = ndimage.median_filter(frame, filtersize)

    processed_frame = frame - frame_smooth

    return processed_frame

start = time.time()

# an example of running this on one image
with fits.open(fileformat.format(ny, nx, 0)) as hdulist:
    data = hdulist[0].data

    filt_data = process_image(data)


single_time = time.time() - start
print('1 image/1 process: {:.2f} seconds'.format(single_time))

#### Activity:
# write and time some code that runs this on all 25 images in parallel. How does the performance increase
# as you increase the number of processes you use?
# we recommend you use multiprocessing pool for this task
if __name__ == "__main__":
    start_2 = time.time()

    pool = mp.Pool(processes=13)

    pool_jobs = []
    for i in range(25):
        with fits.open(fileformat.format(ny, nx, i)) as hdulist:
            data = hdulist[0].data
            task = pool.apply_async(process_image, (data,))

        pool_jobs.append(task)
        print("Created job {0}".format(i))

    for i, task in enumerate(pool_jobs):
        result = task.get()
        print("Job result {0}. First value is {1}".format(i, result.ravel()[0]))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    single_time_2 = time.time() - start_2
    print('25 image/13 process: {:.2f} seconds'.format(single_time_2))

