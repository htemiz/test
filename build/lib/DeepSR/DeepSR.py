"""
DeepSR.py

An open source framework for implementation of the tasks regarding Super Resolution
with Deep Learning architectures based on Keras framework.

The main goal of this framework is to empower users/researchers to focus on their studies
while pursuin g successful Deep Learning algorithms for the task of Super Resolution,
saving them from the workloads of programming, implementing and testing.

It offers several ways to use the framework, such as using the command line, using the DeepSR
object in another program, or using batch files to automate successive multiple jobs.
The DeepSR governs each steps in the workflow of Super Resolution task, such as pre-processing,
augmentation, normalization, training, post-processing, and testing, in such a simple, easy and
fast way that there would remain only a very small amount of work to users/researchers to
accomplish their experiments.


Developed  by
        Hakan Temiz             htemiz@artvin.edu.tr
        Hasan Şakir Bilge       bilge@gazi.edu.tr

Version : 1.200
History :

"""


from sys import argv
import numpy as np
from DeepSR import utils
from DeepSR.args import getArgs
ARGS = getArgs()
if ARGS['seed'] is not None:
    np.random.seed(ARGS['seed'])
    from tensorflow import set_random_seed
    set_random_seed(ARGS['seed'])

if ARGS['backend'] is not None:
    from os import environ
    environ['KERAS_BACKEND'] = ARGS['backend']

import h5py
import time
import pandas as pd
from os import  makedirs, rename
from os.path import isfile, join, exists, splitext, abspath, basename, dirname, isdir
from skimage.io import imread
from matplotlib import pyplot as plt
from PIL import Image
from importlib import import_module
from tqdm import tqdm

from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.backend import image_dim_ordering

image_extentions = ["*.bmp", "*.BMP", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.TIFF", "*.tiff"]


class DeepSR(object):

    def __init__(self, args = None):
        """ takes arguments via a dictionary object and generate class members

        """
        self.__version__ = '1.103'

        if args is None:
            return
        else:
            self.set_settings(args)


    def calculate_mean_of_images(self, imagedir, separate_channels=True, direction='both' ):
        """
        Calculates the mean of images in a folder.
        :param imagedir: Folder containing images.
        :param separate_channels: Indicates that the mean values are separately calculated fro each color channnels.
        Default is True.
        :param direction: Calculation style of mean values. Use 'column' to calculate mean values of each column separately.
        Use 'row' to calculate the mean values of each rows separately. Use 'both' for calculation without separating rows or columns.
        :return: Returns the mean value of image(s).
        """

        files = utils.GetFileList(imagedir, image_extentions, remove_extension=False, full_path=True)

        total_images =0.0
        total_mean =0.0

        for f in files:
            if not isfile(f):
                pass
            else:
                img = imread(f, flatten=False, mode=self.colormode)
                mean = utils.mean_image(img, separate_channels= separate_channels, direction=direction)
                total_mean += mean
                total_images +=1
        print(total_mean, total_images)
        print('Mean : ', total_mean / total_images)
        return total_mean / total_images


    def get_number_of_samples(self, imagedir):
        """
        Gives the number of sub sample images to be produced from the images.
        :param imagedir: Path to the directory containing image(s).
        :return: The number of sub images in total
        """
        total_num = 0 # number of sub images in total

        files = utils.GetFileList(imagedir, image_extentions, remove_extension=False, full_path=True)

        for f in files:
            if not isfile(f):
                print('Warning! the file ', f, '  is not a valid file!')
                continue

            img = imread(f, mode=self.colormode)
            for orijinal_image in utils.augment_image(img, self.augment):

                row, col = img.shape[0:2]
                # we presume remains cropped from the image
                row = int(row / self.scale)
                col = int(col / self.scale)

                if self.upscaleimage:
                    row *= self.scale
                    col *= self.scale

                ver_num = int((row - self.inputsize) // self.stride) + 1
                hor_num = int((col - self.inputsize) // self.stride) + 1

                total_num += ver_num * hor_num

        return total_num


    def generate_batch_on_fly(self, imagedir, shuffle = True):
        """
        Generates asinput and reference (ground-truth) image patches from images in a folder
        for every batches in training procedure.
        :param imagedir: path to directory where images are in
        :param shuffle: Images in batch will be shuffled if it is True.
        :return: Yields two lists composed of as many input and reference image patches as the number of batch.
        """

        lr_stride = self.stride


        # "size" variable is used for calculation of the left, right, top and bottom
        # coordinates of the output patch of ground truth image.
        # if the model's layers uses border effect set to "valid", output image would be
        # smaller than the input image. Hence, the sub patch of ground truth image, relevant to
        # the sub patch of the input image must be the located at the same region and size.
        count = 0
        # iteration = 0
        #print("\nIteration : ", iteration, "\n")

        if self.upscaleimage:
            pad = int((self.inputsize - self.outputsize) / 2)
            size = self.inputsize
            hr_stride = lr_stride
        else:
            pad = int((self.inputsize * self.scale - self.outputsize) / 2)
            size = self.outputsize
            hr_stride = self.scale * lr_stride


        if image_dim_ordering() == "tf":
            input_list = np.empty((self.batchsize, self.inputsize, self.inputsize, self.channels))
            output_list = np.empty((self.batchsize, self.outputsize, self.outputsize, self.channels))
        else:
            input_list = np.empty((self.batchsize, self.channels, self.inputsize, self.inputsize))
            output_list = np.empty((self.batchsize, self.channels, self.outputsize, self.outputsize))

        files = utils.GetFileList(imagedir, image_extentions, remove_extension=False, full_path=True)

        while True:

            if shuffle: # shuffle the file orders
                np.random.shuffle(files)

            for i in tqdm(range(len(files))):
                f = files[i]

                if not isfile(f):
                    continue

                img = imread(f, flatten=False, mode=self.colormode)

                # if channel is set 3 and the input image is a single channel image,
                # image to be converted to 3 channel image copying itself along the third
                # axis
                if len(img.shape) < 3 and self.channels ==3:
                    img = self.single_to_3channel(img, f)

                augmented_images =  utils.augment_image(img, self.augment)

                if shuffle:
                    np.random.shuffle(augmented_images)

                for orijinal_image in augmented_images:

                    # prepare ground truth image
                    img_ground = utils.Preprocess_Image(orijinal_image, scale=self.scale, pad=0,
                                                        channels=self.channels, upscale=False,
                                                        crop_remains=True, img_type='ground')

                    if self.normalizeground: # normalize the ground truth image if it is set to True
                        img_ground = self.normalize_image(img_ground)

                    # prepare input image
                    img_input = utils.Preprocess_Image(orijinal_image, scale=self.scale, pad=0,
                                                       channels=self.channels, crop_remains=True,
                                                       upscale=self.upscaleimage)
                    #
                    # Normalize image for processing
                    #
                    if self.normalize:
                        img_input = self.normalize_image(img_input)

                    # how many sub pictures we can generate
                    ver_num = int((img_input.shape[0] - self.inputsize) // self.stride) + 1
                    hor_num = int((img_input.shape[1] - self.inputsize) // self.stride) + 1

                    # to get sub images in random order, we make a list
                    # and then, shuffle it
                    ver_num_list = list(range(0,ver_num))
                    hor_num_list = list(range(0,hor_num))
                    np.random.shuffle(ver_num_list)
                    np.random.shuffle(hor_num_list)

                    for i in ver_num_list:
                        for j in hor_num_list:

                            lrow_start = i * lr_stride
                            lcol_start = j * lr_stride
                            ihr_stride = i * hr_stride
                            jhr_stride = j * hr_stride

                            hrow_start= ihr_stride + pad
                            hrow_stop = ihr_stride + size - pad
                            hcol_start = jhr_stride + pad
                            hcol_stop = jhr_stride + size - pad

                            # sub patch of input image
                            sub_img = img_input[lrow_start: lrow_start + self.inputsize,
                                      lcol_start: lcol_start + self.inputsize]
                            # reference (ground truth) image patch
                            sub_img_label = img_ground[hrow_start: hrow_stop, hcol_start: hcol_stop]

                            if image_dim_ordering() =="tf": # tensorflow backend
                                sub_img = sub_img.reshape([1, self.inputsize, self.inputsize, self.channels])
                                sub_img_label = sub_img_label.reshape([1, self.outputsize, self.outputsize, self.channels])
                            else:# theano backend
                                sub_img = sub_img.reshape([1, self.channels, self.inputsize, self.inputsize])
                                sub_img_label = sub_img_label.reshape([1, self.channels, self.outputsize, self.outputsize])

                            input_list[count]  = sub_img
                            output_list[count] = sub_img_label

                            if count == self.batchsize -1:
                                yield input_list, output_list
                                count = 0
                            else:
                                count += 1


    def generate_from_hdf5(self, filepath, batchsize, shuffle=False):
        """
        Generates training data in batches from training image patches stored in a .h5 file.
        :param filepath: The path to folder containing input images.
        :param batchsize: The number of image samples for a batch.
        :param shuffle: The images will be shuffled if it is True.
        :return: Two lists composed of training data (input images and corresponding reference images).
        """
        f = h5py.File(filepath, "r")
        rows = f['input'].shape[0]
        indexes = np.arange((rows // batchsize) * batchsize)
        f.close()

        while 1:
            f = h5py.File(filepath, "r")
            if shuffle:
                np.random.shuffle(indexes)

            # count how many entries we have read
            n_entries = 0
            # as long as we haven't read all entries from the file: keep reading
            while n_entries < (rows - batchsize):
                if shuffle:
                    i = indexes[n_entries: n_entries + batchsize]
                    i = np.sort(i).tolist()

                    xs = f['input'][i]
                    ys = f['label'][i]
                else:
                    xs = f['input'][n_entries: n_entries + batchsize]
                    ys = f['label'][n_entries:n_entries + batchsize]

                n_entries += batchsize
                yield (xs, ys)
            f.close()


    def generate_from_hdf5_without_shuffle(self, filepath, batchsize):
        """
        Generates training data in batches from training image patches stored in a .h5 file.
        It is the same as the method 'generate_from_hdf5' except that this method does not shuffle the images.
        :param filepath: The path to folder containing input images.
        :param batchsize: The number of image samples for a batch.
        :return: Two lists composed of training data (input images and corresponding reference images).
        """

        f = h5py.File(filepath, "r")
        rows = f['input'].shape[0]

        f.close()

        while 1:

            f = h5py.File(filepath, "r")

            # count how many entries we have read
            n_entries = 0
            # as long as we haven't read all entries from the file: keep reading
            while n_entries < (rows - batchsize):
                xs = f['input'][n_entries: n_entries + batchsize]
                ys = f['label'][n_entries:n_entries + batchsize]

                n_entries += batchsize
                yield (xs, ys)

            f.close()


    def get_backend_shape(self, input_size=None):
        """
        Forms a tuple object having the same dimensioning style  as the keras backends: Theano or Tensorflow.
        For example, Tensorflow uses following image shape (input_size, input_size, channels), while
        Theano uses (channels, input_size, input_size).
        :param input_size: The size of image. The same value is used for both width and height.
        :return: A tuple object having the shape format as Keras backend.
        """

        if image_dim_ordering() == "tf":  # tensorflow style input shape
            shape = (input_size, input_size, self.channels)
        else:  # theano style input shape
            shape = (self.channels, input_size, input_size)

        return shape


    def get_mean_and_deviation(self):
        """
        Gives the mean and deviation values from the training images.
        :return: The mean and standard deviation values.
        """

        files = utils.GetFileList(self.traindir, image_extentions, remove_extension=False, full_path=True)
        result = np.ones((len(files), 2,self.channels), np.float64)

        for i in range(len(files)):

            f = files[i]
            if not isfile(f):
                print('Warning! the file ', f, '  is not a valid file!')
                continue

            img = imread(f, mode=self.colormode)
            if len(img.shape) == 2: # single channel image
                img = img.reshape(-1,1)
            else: # multi channel image
                img = img.reshape(-1, img.shape[-1])

            result[i] = np.array([img.mean(axis=0), img.std(axis=0)])
        r  = result.mean(axis=0)

        self.mean = r[0]
        self.std  = r[1]

        return self.mean, self.std


    def model_exist(self):
        """
        Check if model exist or not.
        :return: False, if model does not exist. True, otherwise
        """
        if self.model is None:  # build_model method not successfull
            print('Model couldn\'t have been created. Please, check whether \n" +'
                  '\'build_model\' method exist and working properly')
            return False

        return True


    def normalize_image(self, img):
        """
        Normalizes given image. The following normalization procedures can be implemented:

        Min-Max, dividing by a value, subtracting the mean, standardization, and mean normalizaiton.

        :param img: The image to be normalization procedure applied.
        :return: Normalized image.
        """
        if self.normalize is None:
            return img

        if self.normalize[0] == 'divide':
                img = img.astype(np.float64) / self.divident

        #
        # MINMAX NORMALIZATION
        # Inew = (I - min) * (newMax - newMin) / (max - min) + newMin
        #
        elif self.normalize[0] == 'minmax':
            minimum = img.min(axis=(0, 1))
            maximum = img.max(axis=(0, 1))

            if self.minimum is not None and self.maximum is not None:

                img = (img - minimum) * (self.maximum - self.minimum) / (maximum - minimum) + self.minimum

                self.oldminimum = minimum
                self.oldmaximum = maximum

            else:
                print("WARNING!\n\tNew minimum and maximum values were not given. 0-1 normalization being applied to the image!")

                img = (img - minimum) / (maximum - minimum)

        elif self.normalize[0] == 'standard':
            if 'single' in self.normalize:
                mean = img.mean(axis=(0,1))
                std = img.std(axis=(0,1))
                img = (img - mean) / std

            elif self.mean is not None and self.std is not None:
                img = (img - self.mean) / self.std

            else: # mean and std values were not given take them from the image
                mean = img.mean(axis=0)
                std = img.std(axis=0)
                img =(img - mean) / std

        elif self.normalize[0] == 'subtractmean':
            if self.mean is not None:
                img = img - self.mean
            else: # mean value is not given. calculate from the image
                self.mean = img.mean(axis=0)
                img = img - self.mean

        elif self.normalize[0] == 'mean':
            if self.mean is not None and self.minimum is not None:
                img -= self.mean

            else:
                print("WARNING!\n\t")


        return img


    def normalize_image_back(self, img):
        """
        This method applies the reverse of normalization procedure to given image.
        :param img: Image to be reverse normalization
        :return: Image that reverse normalized.
        """
        if self.normalize is None:
            return img

        if self.normalize[0] == 'divide':
                img = np.uint8(img * self.divident)

        elif self.normalize[0] == 'minmax':
            pass

        elif self.normalize[0] == 'standard':
            if 'single' in self.normalize:
                print('WARNING! \n\tSince standardization type is set to \"single\", image can not be normalized back! \
                      \n\tImage returned back intact  without any processing!')

            elif self.mean is not None and self.std is not None:
                img = np.uint8(img * self.std  + self.mean)

            else: # ambiguous situation
                print('WARNING!\n\tAmbiguous situation has occurred since standardization method is not clear!')

        elif self.normalize[0] == 'mean':
            if self.mean is not None:
                img += self.mean

            else: # mean value is not given. calculate from the image
                print('WARNING! \n\tSince mean value is not known, image can not be normalized back with mean value! \
                                      \n\tImage returned back without any processing!')

        return img


    def plotimage(self, im_input, weight_file, plot=True):
        """
        Plots the output image of model along with reference image and interpolation image.
        :param im_input: Input image.
        :param weight_file: weight file of model.
        :param plot: Images to be plotted if it is True.
        :return: Result image along with PSNR and SSIM measures.
        """

        self.model = self.build_model(self, testmode=True)
        if not self.model_exist():  # check if model exists
            print('!ERROR: model not exist! Please, build SR with a model!')
            return None, None, None

        img = imread(im_input, flatten=False, mode=self.colormode)


        # prepare ground image
        img_ground = utils.Preprocess_Image(img, scale=self.scale, pad=self.crop, channels=3,
                                            upscale=False, crop_remains=True, img_type='ground')
        # prepare bicubic image
        img_bicubic = utils.Preprocess_Image(img, scale=self.scale, pad=self.crop, channels=3,
                                             upscale=True, crop_remains=True, img_type='bicubic')

        # prepare low resolution image
        img_input = utils.Preprocess_Image(img, scale=self.scale, pad=0, channels=self.channels,
                                           upscale=self.upscaleimage, crop_remains=True, img_type='input')

        img_input = utils.Prepare_Image_For_Model_Input(img_input, self.channels, image_dim_ordering())

        # Prediction
        img_result = self.predict(img_input, weight_file)

        # PSNR and SSIM
        psnr_model, ssim_model = \
            utils.calc_metrics_of_image(img_ground, img_result, self.crop_test, self.cchannels)

        # if one channel is used only , other channels gathered from bicubic image
        if self.channels == 1:
            im_tmp = img_bicubic
            im_tmp[:, :, 0] = img_result[:, :]
            img_result = im_tmp

        if self.colormode != self.target_cmode:
            if plot:
                img_ground = Image.fromarray(img_ground, self.colormode).convert(self.target_cmode)
                img_bicubic = Image.fromarray(img_bicubic, self.colormode).convert(self.target_cmode)

                img_result = Image.fromarray(img_result, self.colormode).convert(self.target_cmode)

        if plot:
            plt.subplot(221)
            plt.imshow(img_ground)
            plt.title('Ground Image')

            plt.subplot(222)
            plt.imshow(img_bicubic)
            plt.title('Bicubic Image')

            ax = plt.subplot(224)
            plt.imshow(img_result)
            plt.title('Output Image')
            plt.text(0, 0, "PSNR: {0:.4f},  SSIM: {1:.4f}".format(psnr_model, ssim_model), color="red",
                     transform=ax.transAxes)

            plt.subplot(223)
            plt.imshow(img_result)
            plt.title('Exact Output Image')

            plt.tight_layout(pad=2, w_pad=4., h_pad=2.0)
            plt.show()

        return img_result, psnr_model, ssim_model


    def plot_all_layer_outputs(self, img_input, name, plot=False, saveimages=False):
        """"
        Plots the outputs of each layer.
        :param img_input: Input image.
        :param name: The name of the layer.
        :param plot: The layer outputs to be plotted if it is True.
        :param saveimages: Image(s) to be saved if it is True.
        :return:
        """

        import keras.backend as K
        print('****  shape  *****')
        print(img_input.shape)

        for i in range(1, len(self.model.layers)):
            lay = self.model.layers[i]
            get_activations = K.function([self.model.layers[0].input, K.learning_phase()], [lay.output, ])
            activations = get_activations([img_input, 0])[0]

            for j in range(activations.shape[3]):
                r = activations[0, :, :, j].copy()
                if plot:
                    plt.imshow(r, cmap='gray')
                    plt.title('Layer {} , feature map {}'.format( lay.name, j+1))
                    plt.tight_layout()
                    plt.show()
                if saveimages:
                    Image.fromarray(r).save(name + '_' + lay.name + '_filter_' + str(j) + '.tif')


    def plot_layer_output(self, im, layer_idx, saveimages=False):
        """
        Plots the output of a particular layer.
        :param im: Input image.
        :param layer_idx: The index number of the layer whose outputs to be plotted.
        :param saveimages:  Image to be saved if it is True.
        :return:
        """

        import keras.backend as K

        im = imread(im)

        img_input = utils.Preprocess_Image(im, scale=self.scale, pad=0,
                                           channels=self.channels, upscale=self.upscaleimage,
                                           crop_remains=True, img_type='input')

        if self.normalize:
            img_input = self.normalize_image(img_input)

        img_input = utils.Prepare_Image_For_Model_Input(img_input, self.channels, image_dim_ordering())

        lay = self.model.layers[layer_idx]

        get_activations = K.function([self.model.layers[0].input, K.learning_phase()], [lay.output, ])
        activations = get_activations([img_input, 0])[0]

        for i in range(activations.shape[3]):
            r = activations[0, :, :, i].copy()
            plt.imshow(r, cmap='gray')
            plt.title('Layer {} , feature map {}'.format( lay.name, i+1))
            plt.tight_layout()
            plt.show()


    def plot_model(self, to_file=True, dpi=600):
        """
        Plots the diagram of the model using Keras functionality.
        :param to_file: Boolean. Saves the model's diagram as a .png file in the output folder.
        :param dpi: Resolution in dpi.
        :return:
        """
        from keras.utils import plot_model
        if to_file:
            plot_model(self.model, to_file=self.outputdir + self.modelname + '.png')
        else:
            plot_model(self.model)
            plt.show()


    def plot_layer_weights(self, saveimages=True, plot=False, name='', dpi=300):
        """
        Plots layer weights on screen and/or as images.
        :param saveimages:Boolean. Layer weights to be saved as images if it is True.
        :param plot: Boolean. Layer weights to be drawn as figures if it is True.
        :param name: A name prefixed before layer names.
        :param dpi:  Resolution in dpi.
        :return:
        """

        border = 1
        for lay in self.model.layers:
            weights = lay.get_weights().copy()

            if len(weights) != 0:

                count=1
                weights = weights[0] # first part of layers, second part is bias of layer

                if  len(weights.shape) == 4:


                    h, w = weights.shape[0], weights.shape[1]
                    nrows, ncols= weights.shape[2], weights.shape[3]
                    nsamp = nrows * ncols
                    weights= weights.reshape(h, w, nsamp).copy()
                    weights= weights.swapaxes(1,2).swapaxes(0,1)

                    mosaic = np.ma.masked_all((nrows * h + (nrows - 1) * border,
                                            ncols * w + (ncols - 1) * border),
                                           dtype=np.float64)
                    paddedh = h + border
                    paddedw = w + border

                    for i in range(nsamp):
                        row = int(np.floor(i / ncols))
                        col = i % ncols

                        mosaic[row * paddedh:row * paddedh + h,
                        col * paddedw:col * paddedw + w] = weights[i,:,:]
                    fig = plt.figure()

                    ax = plt.subplot()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    im = ax.imshow(mosaic, interpolation=None, cmap='gray')
                    ax.set_title(
                        "Layer '{}' of type {}".format(lay.name, lay.__class__.__name__))

                    fig.tight_layout()

                    if plot :
                        plt.show()

                    if saveimages :
                        ax.figure.savefig(name + '_layer_' + lay.name + '_' + str(count)+ '.png', dpi=dpi)

                    count +=1


    def predict(self, img_input, weight_file, normalizeback = False):
        """
        Returns the output image of the model.
        :param img_input: Input image
        :param weight_file: Weight file to load the model with.
        :param normalizeback: Boolean. Reverse normalization to be applied to the output image if it is True.
        :return:
        """

        if not self.model_exist():  # check if model exists
            print("ERROR! Model not exists!")
            return None

        self.model.load_weights(weight_file)

        img_result = self.model.predict(img_input, batch_size=1)
        if image_dim_ordering() == "tf":  # tensorflow backend
            if self.channels == 1:
                img_result = img_result[0, :, :, 0]
            else:
                img_result = img_result[0, :, :, 0:self.channels]
        else:  # theano backend
            if self.channels == 1:
                img_result = img_result[0, 0, :, :]
            else:
                img_result = img_result[0, self.channels, :, :]

            # since THEANO has image channels first in order,
            # channels need to put in last in order (w,h, channels)
            img_result = img_result.transpose((1, 2, 0))


        # normalize image back if normalization is applied.
        # if normalization is applied and it is necessary to reverse the normalization back.
        if normalizeback and self.normalize is not None :  # normalize the input image if it is set to True
            img_result = self.normalize_image_back(img_result)
            return img_result

        else:
            return img_result.astype(np.uint8)


    def prepare_dataset(self, imagedir=None, datafile=None):
        """
        Constructs a dataset as a .h5 file containing input and corresponding reference images for training.
        :param imagedir: Path to the folder containing training images.
        :param datafile: Path to the output file.
        :return:
        """

        if not exists(self.datadir):
            makedirs(self.datadir)

        if datafile is None:
            datafile = self.datafile

        if imagedir is None:
            imagedir = self.traindir

        chunks = 3192
        input_nums = 1024
        lr_stride = self.stride

        if self.upscaleimage:
            hr_stride = self.scale * lr_stride
        else:
            hr_stride = lr_stride


        with h5py.File(datafile, 'w') as hf:
            if image_dim_ordering() == "tf":  # tensorflow backend
                hf.create_dataset("input", (input_nums, self.inputsize, self.inputsize, self.channels),
                                 maxshape=(None, self.inputsize, self.inputsize, self.channels),
                                 chunks=(self.batchsize, self.inputsize, self.inputsize, self.channels),
                                 dtype='float32')
                hf.create_dataset("label", (input_nums, self.outputsize, self.outputsize, self.channels),
                                 maxshape=(None, self.outputsize, self.outputsize, self.channels),
                                 chunks=(self.batchsize, self.outputsize, self.outputsize, self.channels),
                                 dtype='float32')
            else:
                hf.create_dataset("input", (input_nums, self.channels, self.inputsize, self.inputsize),
                                 maxshape=(None, self.channels, self.inputsize, self.inputsize),
                                 chunks=(128, self.channels, self.inputsize, self.inputsize),
                                 dtype='float32')
                hf.create_dataset("label", (input_nums, self.channels, self.outputsize, self.outputsize),
                                 maxshape=(None, self.channels, self.outputsize, self.outputsize),
                                 chunks=(128, self.channels, self.outputsize, self.outputsize),
                                 dtype='float32')

        count = 0

        files = utils.GetFileList(imagedir, image_extentions, remove_extension=False, full_path=True)

        for f in files:
            if not isfile(f):
                continue
            print(f)

            img = imread(f, flatten=False, mode=self.colormode)

            for ref_image in utils.augment_image(img, self.augment):

                w, h, c = ref_image.shape
                w -= int(w % self.scale)  # exact fold of the scale
                h -= int(h % self.scale)  # exact fold of the scale

                # -
                # -
                # -------  MODIFY THIS SECTION SINCE NORMALIZATION METHOD WAS CHANGED ---
                # -
                # -

                # prepare ground truth, input and bicubic images
                img_ground, img_bicubic, img_input = \
                    utils.Prepare_Input_Ground_Bicubic(ref_image, scale=self.scale, pad=self.crop,
                                                       upscaleimage=self.upscaleimage, channels=self.channels)
                # -
                # -
                # -

                # how many sub pictures we can generate
                ver_num = int((h - self.outputsize) / hr_stride)
                hor_num = int((w - self.outputsize) / hr_stride)

                h5f = h5py.File(datafile, 'a')

                if count + chunks > h5f['input'].shape[0]:
                    input_nums = count + chunks

                    if image_dim_ordering() == "tf":  # tensorflow style image ordering
                        h5f['input'].resize((input_nums, self.inputsize, self.inputsize, self.channels))
                        h5f['label'].resize((input_nums, self.outputsize, self.outputsize, self.channels))
                    else:  # theano style image ordering
                        h5f['input'].resize((input_nums, self.channels, self.inputsize, self.inputsize))
                        h5f['label'].resize((input_nums, self.channels, self.outputsize, self.outputsize))

                for i in range(0, hor_num):
                    for j in range(0, ver_num):
                        lrow_start = i * lr_stride
                        lcol_start = j * lr_stride

                        sub_img = img_input[lrow_start: lrow_start + self.inputsize,
                                  lcol_start: lcol_start + self.inputsize]
                        if image_dim_ordering() == "tf":  # tensorflow backend
                            sub_img = sub_img.reshape([1, self.inputsize, self.inputsize, self.channels])
                        else:  # theano backend
                            sub_img = sub_img.reshape([1, self.channels, self.inputsize, self.inputsize])

                        ihr_stride = i * hr_stride
                        jhr_stride = j * hr_stride
                        sub_img_label = img_ground[ihr_stride + self.crop: ihr_stride + self.outputsize,
                                        jhr_stride + self.crop: jhr_stride + self.outputsize]

                        if image_dim_ordering() == "tf":  # tensorflow backend
                            sub_img_label = sub_img_label.reshape([1, self.outputsize, self.outputsize, self.channels])
                        else:
                            sub_img_label = sub_img_label.reshape([1, self.channels, self.outputsize, self.outputsize])

                        h5f['input'][count] = sub_img
                        h5f['label'][count] = sub_img_label
                        count += 1

        if image_dim_ordering() == "tf":  # tensorflow backend
            h5f['input'].resize((count, self.inputsize, self.inputsize, self.channels))
            h5f['label'].resize((count, self.outputsize, self.outputsize, self.channels))
        else:  # THENAO style image ordering
            h5f['input'].resize((count, self.channels, self.inputsize, self.inputsize))
            h5f['label'].resize((count, self.channels, self.outputsize, self.outputsize))

        h5f.close()


    def prepare_delegates(self):
        """
        Builds callback delegates for Keras model.
        :return:
        """

        callbacks = []
        path = join(self.outputdir, 'weights.{epoch:02d}.h5')

        # If there is not any validation image(s), so loss method should
        # be 'loss', 'val_loss', otherwise
        if self.valdir is None or self.valdir == '':
            loss = 'loss'
        else:
            loss = 'val_loss'

        model_checkpoint = ModelCheckpoint(path, monitor=loss, save_best_only=False,
                                           mode='min', save_weights_only=False)

        callbacks.append(model_checkpoint)
        callbacks.append(CSVLogger(self.outputdir + '/training_' + image_dim_ordering() + '_.log'))
        callbacks.append(EarlyStopping(monitor=loss, patience=self.earlystoppingpatience, verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor=loss, factor=self.lrateplateaufactor, patience=self.earlystoppingpatience,
                                           verbose=1, mode='min', min_lr=self.minimumlrate))
        return callbacks


    def print_weights(self):
        """
        Prints layer weights in command prompt.
        :return:
        """

        for layer in self.model.layers:
            g = layer.get_config()
            h = layer.get_weights()
            print(g)
            print(h)


    def repeat_test(self, count):
        """
        Repeats the test procedure only for one epoch at a time.
        :param count: The number repeats.
        :return:
        """

        scale = str(self.scale)
        csv_file= self.outputdir + "/results.csv"

        for i in range(0, count):
            print("\n*** ITERATION %03d ***" % (i+1))
            result_folder = abspath(join(self.outputdir, 'repeat_test', str(i + 1)))
            if not exists(result_folder):
                utils.makedirs(result_folder)

            self.set_model(self.build_model)

            self.train_on_fly()
            _im, d = self.test()
            df = None
            if i == 0:
                df = pd.DataFrame(d[scale].loc['Mean']).transpose()
                df.rename(index={'Mean': str(i + 1)}, inplace=True)
                results = df
            else:
                df = pd.DataFrame(d[scale].loc['Mean']).transpose()
                df.rename(index={'Mean': str(i + 1)}, inplace=True)
                results = results.append(df)

            files = utils.GetFileList(self.outputdir, ["*.h5", '*.xlsx', '*.txt', '*.log'], remove_extension=False,
                                      full_path=False)
            for f in files:
                rename(abspath(join(self.outputdir, f)), result_folder + "/" + f)

        min = results.min()
        mean = results.mean()
        max = results.max()

        results.loc['Min'] = min
        results.loc['Mean'] = mean
        results.loc['Max'] = max

        results.to_csv(csv_file, sep=';')

        # since Excel uses comma (,) for number dot
        t = ''
        with open(csv_file, 'r') as f:
            t = f.read()

        with open(csv_file, 'w') as f:
            t = t.replace('.', ',')
            f.write(t)


    def save_history(self, history):
        """
        Stores the output values in a file after each epoch.
        :param history:
        :return:
        """
        length = len(history.history)
        try:
            f = open(self.outputdir + "/history.txt", 'w')
            for k in history.history.keys():

                f.write(str(k) + "; ")

            for i in range(0, length):
                text= "\r\n"

                for v in history.history.values():
                    text += str(v[i]) + "; "
                f.write(text)

            f.close()
        except:
            return False


    def set_model(self, build_fn):
        """
        Builds the model for training or test procedures.
        :param build_fn: A method returns a Keras model.
        :return:
        """
        self.build_model = build_fn
        if hasattr(self, 'mode') and self.mode == "test":
            self.model = self.build_model(self, testmode=True)
            print("Model has been created for TEST")
        else:
            self.model = self.build_model(self, utils, testmode=False)
            print('Model has been created for TRAINING')


    def set_settings(self, args=None):
        """
        Initials the DeepSR object.
        :param args: Command arguments.
        :return:
        """

        args_has_build_function=False

        if args is None:
            print("There is no any parameters given. Class constructed with no any parameters")
            return

        command_parameters = ["train", "test", 'predict', "repeat_test", 'shutdown', 'plotimage']

        # parameters passed via command line has priority to the parameters in model file
        # so override the parameters that prohibited in command line
        for key in args.keys():
            if args[key] != None and key not in command_parameters:

                setattr(self, key, args[key])


        if  'model' in args and args['model'] != "" :
            from sys import path # necessary for importing the module
            path.append(dirname(args['model']))
            model_name = basename(args['model']).split('.')[0]
            module = import_module(model_name, args["model"])
            self.build_model = module.build_model
            args_has_build_function = True

            # take setting from settings dictionary that exist in the module
            for key in module.settings.keys():
                setattr(self, key, module.settings[key])
        else:
            print("Warning! SRBase Class created with no any model! Please set a model before training or testing\n" +
                  "Or, reconstruct the class with parameters in command line " +
                  " or with a dictionary consists of parameters.\n" +
                  "Refer to args.py file or Readme.txt for information and instructions")

        for key in args.keys():
            if ('--'+key) in argv:
                if args[key] != None and args[key] != "" and key not in command_parameters:

                    setattr(self, key, args[key])



        if self.workingdir == "":  # working directory is not set
            from os import getcwd  # current folder is the working folder
            self.workingdir = getcwd()

        if not hasattr(self, 'outputdir') or self.outputdir == "":
            self.outputdir = abspath(self.workingdir + '/' + self.modelname + "/output/" + str(self.scale))
            if not exists(self.outputdir):
                makedirs(self.outputdir)

        if not hasattr(self, "datadir") or self.datadir =="":
            self.datadir = abspath(self.workingdir + '/' + self.modelname + '/data')
            if not exists(self.datadir):
                makedirs(self.datadir)

        if not hasattr(self, "target_cmode") or self.target_cmode == "":
            self.target_cmode = self.colormode

        #
        # set the OUTPUTSIZE #
        if self.upscaleimage:
            self.outputsize = self.inputsize - 2 * self.crop # if pad value exist, subtract from the outputsize
        else:
            self.outputsize = self.inputsize * self.scale

        # prepare file names #
        data_file = "training_" + image_dim_ordering() + "_" + str(self.scale) + "_" + \
                    str(self.inputsize) + "_" + str(self.outputsize) + \
                    "_" + str(self.stride) + ".h5"

        self.datafile = join(self.datadir, data_file)

        validation_file  = "validation_" + image_dim_ordering() + "_" + str(self.scale) + "_" + \
                    str(self.inputsize) + "_" + str(self.outputsize) + \
                    "_" + str(self.stride) + ".h5"

        self.validationfile = join(self.datadir, validation_file)

        if args_has_build_function:
            self.model = self.build_model( self, testmode=True)

        #
        # Find the normalization method and set relevant values for processing
        #
        if hasattr(self, 'normalize') and self.normalize !='' or \
                self.normalize != False or self.normalize is not None :

            if self.normalize[0] == 'standard':
                if 'whole' in self.normalize: # means that mean and standard deviation to be calculated from training set
                    print("Calculating the mean and the standard deviation from the training set...")
                    self.get_mean_and_deviation()
                    print("Done!")
                elif 'single' in self.normalize:
                    self.mean = None
                    self.std = None
                elif len(self.normalize) == 3:
                    self.mean = self.normalize[1]
                    self.std = self.normalize[2]
                else:
                    print("WARNING: \n\tSome parameters of standardization is missing. \"single\" parameter is set in order to \
                     \n\tcalculate mean and standard deviation from each images individually to continue.")
                    self.normalize.append('single')
                    self.mean = None
                    self.std = None

            elif self.normalize[0] == 'divide': # divide each image with a value

                if len(self.normalize) == 2:
                    self.divident = float(self.normalize[1])
                else: # divident is not given. take it as 255.0
                    self.divident = 255.0
                    print("WARNING : \n\tThe divident value of division normalization is not given. The divident is set as 255.0")

            elif self.normalize[0] == 'minmax': # normaliza image with minmax normalization method

                if len(self.normalize) == 1: # not any minimum or maximum value was given set them as 0 and 1 respectivelly.
                    print("WARNING : \n\tNeither of minimum or maximum value was given. The are set to 0 and 1, respectivelly!")
                    self.minimum = 0.0
                    self.maximum = 1.0
                else:
                    self.minimum = float(self.normalize[1])
                    self.maximum = float(self.normalize[2])

            elif self.normalize[0] == 'mean': # subtract the mean from images

                if 'whole' in self.normalize: # calculate mean value from training set
                    print("Calculating the mean from the training set...")
                    self.get_mean_and_deviation()
                    print('Done!')
                elif 'single' in self.normalize: # means that each image is processed by subtracting its mean value
                    self.mean = None

                elif len(self.normalize) == 2: # mean value is not given, calculate from training set
                    self.mean = float(self.normalizaiton[1])

                else:
                    print("WARNING! \n\tThe mean value is not provided! It is going to be calculated from the training set.")
                    print("Calculating the mean from the training set...")
                    self.get_mean_and_deviation()
                    print('Done!')

        else:
            print("WARNING! \n\tThere is no any Normalization method. Images to be processed without normalization")
            self.normalize = False

        if len(self.metrics) == 1 and self.metrics[0].upper() == 'ALL':
            self.metrics = utils.METRICS.copy()

        if self.interpmethod is not None and 'ALL' in [x.upper() for x in self.interpmethod]:
            if 'same' in self.interpmethod:
                self.interpmethod = utils.METHODS.copy()
                self.interpmethod.append('same')
            else:
                self.interpmethod = utils.METHODS


    def single_to_3channel(self, img, file_name):
        """
        Converts a single channel image to 3-channel image
        :param img:
        :param file_name: The name of image file.
        :return: 3-channel image.
        """

        tmp = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            tmp[:, :, i] = img[:, :]  # make single channel image to 3 channel image

        print("WARNING! ", file_name, "\n\t\t  is a single-channel image. It is converted ",
              "to 3-channel image copying along the third axis.")

        return tmp


    def test(self, testpath=None, weightpath=None, saveimages=False):

        """
        Tests the model with given test image(s) over given weight file(s). Takes single or multiple
        image(s) and weight file(s).The paths of multiple images or multiple weight files must be a folder path.

        Returns the output image, psnr and ssim measures of given image. The output image is the last
        image processed by the model with the last weight file, in case the 'testpath' parameter
        is a folder path.

        :param testpath: Path to the test image or the folder contains the image(s)
        :param weightpath:  Path to the weight file (.h5 file) or to the folder contains the weight file(s)
        :param saveimages: Boolean. If True, the images of ground truth, bicubic and the result of the model are saved.
        :return: Output image, PSNR measure, SSIM measure

        """
        print("\n[START TEST]")

        self.model = self.build_model(self, testmode=True)
        if not self.model_exist():  # check if model exists
            return None, None, None

        # if 'testpath' is not given during method call, take it from the class
        if testpath == None:
            testpath = self.testpath

        # if 'weightpath' is not given during method call, take it from the class
        if weightpath == None or weightpath == "":
            weightpath = weightfolder = self.outputdir

        # if weightpath is a path to a weight file (.h5 file) 'weightfolder'
        # is set to folder path of weight file
        elif not isdir(weightpath):
            weightfolder = abspath(dirname(weightpath))

        else:
            weightfolder = weightpath

        text_scale = str(self.scale)

        if not isinstance(testpath, (list, tuple)):  # make it list or tuple
            testpath = [testpath]

        if saveimages:
            output_images_dir = join(self.outputdir, 'images')
            if not exists(output_images_dir):
                makedirs(output_images_dir)

        for path in testpath:

            weights_list = {}  # keeps the list of weights for each scale
            print(path)

            test_name = path.split('\\')[-1]

            if isdir(path):
                # veri klasorunde kaç dosya var?
                test_files = utils.GetFileList(path, image_extentions, remove_extension=False, full_path=True)
                test_file_names = [splitext(basename(x))[0] for x in test_files]

            else:
                test_files = list([path])
                test_file_names = splitext(basename(test_name))[0]
                test_file_names = list([test_file_names])

            if isdir(weightpath):
                weights_list = utils.GetFileList(weightfolder, ["*.h5"], remove_extension=False, full_path=True)

            else:
                weights_list = [weightpath]  # weightpath is a file. Should be in a list for iteration.

            if len(weights_list) == 0: # there is no any weight file
                print('Any weight file could not be found in the following path:\n', weightpath, '\nTerminating...')
                return None

            if  self.interpmethod is None or self.interpmethod == '':
                columns = list()

            else:
                columns = self.interpmethod.copy()
                if 'same' in columns:
                    columns.remove('same')

            for w in weights_list:
                columns.append(splitext(basename(w))[0])

            data_columns = columns.copy()

            columns = pd.MultiIndex.from_product([data_columns, self.metrics])
            dataset = pd.DataFrame(index=test_file_names, columns=columns)

            # for each test images
            for i in tqdm(range(len(test_files))):
                f = test_files[i]

                file_short_name = splitext(basename(f))[0]
                #print(f)
                satir = test_file_names[i]

                _im = imread(f, mode=self.colormode)

                # if channel is set 3 and the input image is a single channel image,
                # image to be converted to 3 channel image copying itself along the third
                # axis
                if len(_im.shape) < 3 and self.channels == 3:
                    _im = self.single_to_3channel(_im, f)

                # prepare ground image
                img_ground = utils.Preprocess_Image(_im, scale=self.scale,
                                                    pad=self.crop, channels=self.channels, upscale=False,
                                                    crop_remains=True, img_type='ground')

                if saveimages:
                    fileName = join(output_images_dir, file_short_name)
                    Image.fromarray(img_ground).save(fileName + '_scale_'+ text_scale + '_ground.png')
                    # Image.fromarray(img_bicubic).save(fileName + '_bicubic.png')

                # pad is set zero since input image should not be cut from borders. But, Ground truth and bicubic
                # images are cut from borders as muuch as pad size in case the model reduces the size of the output
                # image as much as pad. Some models work like this to avoid border effect in calculation of metrics.
                # If the parameter of padding is set to 'valid' in layers of models in KERAS is set as , output size
                # of images are decreases as floor integer value of kernel's half width at each layers. Therefore,
                # to have the same size of ground truth, bicubic and output images, bicubic and ground truth images
                # are cropped from borders as much as padding value.
                #
                # The padding value is set to 'same' in keras layers as to have the same size of output and input
                # input images. If this is the case, padding is not applied in here since they are of the same size
                #
                img_input = utils.Preprocess_Image(_im, scale=self.scale, pad=0,
                                                   channels=self.channels, upscale=self.upscaleimage,
                                                   crop_remains=True, img_type='input')

                if self.normalize:
                    img_input = self.normalize_image(img_input)

                img_input = utils.Prepare_Image_For_Model_Input(img_input, self.channels, image_dim_ordering())

                # if any interpolation method is defined, do the test for
                # them
                if self.interpmethod is not None and (len(self.interpmethod) > 0 or self.interpmethod != ''):

                    for method in self.interpmethod:
                        # if interpolation method for upscaling (interpmethod) is
                        # set as 'same', then low resolution image upscaled by the
                        # same interpolation method as decimation.

                        if method == 'same':
                            continue

                        if 'same' in self.interpmethod:
                            im_m = utils.Preprocess_Image(_im,
                                                          scale=self.scale, pad=self.crop, channels=self.channels,
                                                          interp_down=method, interp_up=method,
                                                          upscale=True, crop_remains=True, img_type='bicubic')
                        else:
                            im_m = utils.Preprocess_Image(_im, scale=self.scale, pad=self.crop,
                                                          interp_down=self.decimation, interp_up=method,
                                                          channels=self.channels, upscale=True, crop_remains=True,
                                                          img_type='bicubic')

                        res = utils.calc_multi_metrics_of_image(img_ground, im_m,
                                                                border=self.crop_test, channels=self.cchannels, metrics=self.metrics)

                        if saveimages:
                            Image.fromarray(im_m).save(fileName + '_scale_'+ text_scale + '_' + method + '.png')


                        for key, value in res.items():
                            dataset.loc[file_short_name, (method, key)] = value

                # do test for each weight file(s)
                for j in range(0, len(weights_list)):
                    sutun = splitext(basename(weights_list[j]))[0]

                    # Prediction
                    img_result = self.predict(img_input, weights_list[j],
                                              self.normalizeback)


                    res = utils.calc_multi_metrics_of_image(img_ground, img_result,
                                                            border=self.crop_test, channels=self.cchannels, metrics=self.metrics)

                    for key, value in res.items():
                        dataset.loc[file_short_name, (sutun, key)] = value

                    if saveimages:
                        Image.fromarray(img_result).save(fileName + '_scale_' +
                                            text_scale + '_Result_' + sutun + '.png')

                    if self.layeroutput:
                        name = join(fileName, 'layer_outputs')

                        if not exists(name):
                            makedirs(name)
                        name = join(name, 'scale_') + text_scale + '_' + \
                                self.modelname + '_' + sutun

                        self.plot_all_layer_outputs(img_input, name, saveimages=self.saveimages, plot=False)

                    if self.layerweights :
                        name = join(fileName, 'layer_weights')

                        if not exists(name):
                            makedirs(name)

                        name = join(name, 'scale_') + text_scale + \
                               self.modelname + '_' + sutun

                        self.plot_layer_weights(saveimages=self.saveimages,name=name, plot=False, dpi=300)

            # write results in an excel file in weight folder
            excel_file = self.modelname + '_' + "test_results_" + \
                             text_scale + '_' + test_name + ".xlsx"

            # utils.write_to_excel(datasets, excel_file)
            dataset.loc["Mean"] = dataset.mean()
            utils.measures_to_excel(dataset, self.outputdir, excel_file)

            print("[TEST FINISHED]")

        return dataset


    def train_with_h5file(self, weightpath = None, plot=False):
        """
        Training procedure with images in a .h5 file.
        :param weightpath: Path to model weight.
        :param plot:
        :return:
        """

        self.model = self.build_model(self, testmode=False)
        if not exists(self.datafile): # check if training data exists
            self.prepare_dataset(self.traindir, self.datafile)

        h5f = h5py.File(self.datafile, 'r')
        X = h5f['input']
        y = h5f['label']

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not a file path. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        print(self.model.summary())

        print("Training starting...")
        start_time = time.time()

        self.history = self.model.fit(X, y, validation_split=0.1, batch_size=self.batchsize,
                                 epochs=self.epoch, verbose=0, shuffle=self.shuffle,
                                 callbacks=self.prepare_delegates())
        elapsed_time = time.time() - start_time

        utils.save_model(self.outputdir, self.model, elapsed_time)

        print("Training has taken %.3f seconds." % (elapsed_time))


    def train_on_batch(self, weihgtpath=None, plot=False):
        """
        Training procedure with batches. To be implemented in future.
        :param weihgtpath:
        :param plot:
        :return:
        """

        pass


    def train(self, weightpath= None, plot=False):
        """
        Training procedure of the method.
        :param weightpath: Path to the weight file to load the model with.
        :param plot:
        :return:
        """

        print("\n[START TRAINING]")
        self.model = self.build_model(self, testmode=False)

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not valid. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        train_samples = self.get_number_of_samples(self.traindir)

        train_steps = int(train_samples // self.batchsize)

        # if there is no any validation image(s), train the model without validation images,
        # otherwise, if exist any, train model with the validation image(s)
        if self.valdir is None or self.valdir == '':
            valData= None
            val_steps = None

        else:
            val_samples = self.get_number_of_samples(self.valdir)
            val_steps = int(val_samples // self.batchsize)
            valData = self.generate_batch_on_fly(self.valdir, shuffle=self.shuffle)


        print(self.model.summary())

        print("Batch generator starting...")
        start_time = time.time()

        self.history = self.model.fit_generator(
                                 self.generate_batch_on_fly(self.traindir, shuffle=self.shuffle),
                validation_data= valData,
                epochs =self.epoch, workers=1,max_queue_size=1, callbacks = self.prepare_delegates(),
                steps_per_epoch=train_steps, validation_steps = val_steps, verbose=2 )

        elapsed_time = time.time() - start_time

        utils.save_model(self.outputdir, self.model, elapsed_time)

        print("Training has taken %.3f seconds." % (elapsed_time))
        print("[TRAIN ON FLY FINISHED]")


    def train_with_fit_generator(self, weightpath = None, plot=False):
        """
        Training procedure with Keras Generator.
        :param weightpath: Path to the weight file to load the model.
        :param plot:
        :return:
        """

        self.model = self.build_model(self, testmode=False)
        if not exists(self.datafile): # check if training data exists
            print("\nTRAIN DATASET")
            self.prepare_dataset(self.traindir, self.datafile)

        f = h5py.File(self.datafile, "r")
        row_number = f['input'].shape[0]
        f.close()
        train_steps = int(row_number // self.batchsize)

        # if there is no any validation image(s), train the model without validation images,
        # otherwise, if exist any, train model with the validation image(s)
        if (self.valdir is None or self.valdir == ''):
            valData= None
            val_steps= None

        elif not exist(self.validationfile) :

            print("\nVALIDATION DATASET")
            self.prepare_dataset(self.valdir, self.validationfile)

            f = h5py.File(self.validationfile, "r")
            row_number = f['input'].shape[0]
            f.close()
            val_steps = int(row_number // self.batchsize)
            valData = self.generate_from_hdf5_without_shuffle(self.validationfile, self.batchsize)

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not a file path. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        #print(self.model.summary())

        print("Training on fit generator starting...")
        start_time = time.time()

        self.history = self.model.fit_generator(
                self.generate_from_hdf5_without_shuffle(self.datafile, self.batchsize), validation_data= valData,
                epochs =self.epoch, workers=1,max_queue_size=128, callbacks = self.prepare_delegates(),
                steps_per_epoch=train_steps, validation_steps = val_steps, verbose=2 )

        elapsed_time = time.time() - start_time

        utils.save_model(self.outputdir, self.model, elapsed_time)

        print("Training has taken %.3f seconds." % (elapsed_time))


def start():
    sr = DeepSR(ARGS)
    sr.plot_model()

    if ARGS['train_with_generator'] is not None and ARGS['train_with_generator']:
        print("[Train With Generator]")
        if sr.weightpath == None or sr.weightpath=="":
            sr.train_with_fit_generator()
        else:
            sr.train_with_fit_generator(sr.weightpath)

    if ARGS['train'] is not None and ARGS['train']:
        sr.mode = 'train'
        sr.train(sr.weightpath)

    if ARGS['test'] is not None and ARGS['test']:
        sr.test(sr.testpath, sr.weightpath, ARGS['saveimages'])

    if ARGS['predict'] is not None and ARGS['predict']:
        print("[Predict Mode]")
        sr.predict(sr.testpath, sr.weightpath)

    if ARGS['repeat_test'] is not None and ARGS['repeat_test']:
        sr.repeat_test(ARGS['repeat_test'])

    if ARGS['plotimage'] is not None and ARGS['plotimage']:
        sr.plotimage(sr.testpath, sr.weightpath, True)

    if ARGS['shutdown'] is not None and ARGS['shutdown']:
        from os import system
        time.sleep(60) # wait for a minute so that computer finalizes processess.
        system('shutdown /' + ARGS['shutdown'])

    return sr


if __name__ == "__main__":
    start()

