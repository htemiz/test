"""
args.py

This is the argument file in which argument lists are defined
for the of DeepSR object (DeepSR.py)


Developed  by
        Hakan Temiz             htemiz@artvin.edu.tr
        Hasan Şakir Bilge       bilge@gazi.edu.tr


Version : 1.200
History :

"""

import argparse


parser = argparse.ArgumentParser(description="A Python framework for Image Super Resolution.")

parser.add_argument('--augment',  nargs='*',
                    help="A list of translation operation of image for image data augmenting (90, 180,270, 'flipud', 'fliplr', 'flipudlr')")

parser.add_argument("--backend", type=str, choices=["theano", "tensorflow"], default='tensorflow',
                    help="Determines Keras backend: 'theano' or 'tensorflow'. Defaults is 'tensorflow'.")

parser.add_argument("--batchsize", type=int, default=256,
                    help="Batch size for model training")

parser.add_argument("--channels", type=int, choices=[1,3], default=3,
                    help="Number of color channels (either, 1 or 3) used by model. Default is 3.")

parser.add_argument("--cchannels", type=int, choices=[1,3], default=3,
                    help="Determines what number of color channels used for test. It should be 1 or \
                    3. If it is 1, only first channels of both images compared in tests.")

parser.add_argument("--colormode", type=str, choices=["RGB", "YCbCr"], default= 'RGB',
                    help="Color space (RGB or YCbCr) in which the model processes the image.")

parser.add_argument("--crop", type=int, default=0,
                    help="Cropping size from Image. It is valid for training phase. Ground truth and/or \
                        interpolated images to be cropped by crop size from each borders. Some models \
                        produce images with less size than the one of input images, since they use padding \
                        value as 'valid' to avoid border effects. Therefore, interpolated and/or Ground Truth \
                        images should be cropped to make their size be the same as the size of output image of model.")

parser.add_argument("--crop_test", type=int, default=0,
                    help="Cropping size from Image. Same as the paramater\'crop\', except that it is used for test phase. \
                            Images (output image as well) to be cropped by crop size from each borders. \
                            Some models produce images with less size than the size of input, since they use \
                             padding value as 'valid' to avoid border effects. Therefore, interpolated and/or \
                            Ground Truth images should be cropped to make their size be the same as the size of \
                             output image of model.")

parser.add_argument("--decay", type=float, default=1e-6,
                    help="Decay value for weights.")

parser.add_argument("--decimation", type=str, choices=["bilinear", "bicubic", 'nearest', 'lanczos', 'same'], default='bicubic',
                    help="The Interpolation method used in decimating the image to get low resolution (downscaled) image). \
                         Images to be decimated with this method and upscaled with the interpolation method given in 'interpmethod'. \
                        s decimate images down with the same method(s) as in parameter \'interpmethod\', Use keyword \'same\'. Hence, \
                          The same method(s) given in command argument \'interpmethod\' to be used for decimation of images down \
                          in producing low resolution images (downscaled), and interpolation them to yield upscaled images. \
                          Default is \'bicubic\'.")

parser.add_argument("--earlystoppingpatience", type=int, default=5,
                    help="The number of training epochs to wait before early stop if the model does not progress \
                         on validation data. this parameter helps avoiding overfitting when training the model.")

parser.add_argument("--epoch", type=int, default=1,
                    help="Number of Epoch(s) for training model.")

parser.add_argument("--inputsize", type=int, default=30,
                    help="The size of input image patch in training mode.")

parser.add_argument("--interpmethod", type=str, default='', nargs='*',
                    help="The interpolation method(s) to which the performance of the model is compared. \
                        the interpolation method(s) (bilinear, bicubic, nearest, lanczos) determined with this argument. \
                         If it is None, the model will not be compared with any interpolation methods. Use \'ALL\' for \
                         comparison the performance of the model with all interpolation methods. Use the keyword \'same\' to \
                        use the same interpolation method(s) in decimating down the image (overrides the method given in \
                        the argument \‘decimation\’) and upscaling back. Default is None.")

parser.add_argument("--layeroutput",  action="store_true", default=False,
                    help="Plots layers if it is given.")

parser.add_argument("--layerweights",  action="store_true", default=False,
                    help="Plots layer weights, if it is given.")

parser.add_argument("--lrate", type=float, default=0.001,
                    help="Learning rate for weight updates")

parser.add_argument('--metrics', default= ['PSNR', 'SSIM'],  nargs='*',
                    help="A list of image quality assessment (IQA) metrics  , i.e., \
                    'PSNR, SSIM, FSIM, MSE, MAD, NRMSE'. OR type 'ALL' \
                    to measure entire metrics defined in the list 'METRICS' in 'utils.py'.\
                     Please refer to variable METRICS in utils.py for entire metrics.")

parser.add_argument("--model", type=str, default='',
                   help="Path to the model file (a .py file). Model file must have a method named construct_model \
                       returning the model object, as well as a dictionary, named settings, which has settings in key/value pairs")

parser.add_argument("--modelname", type=str, default='',
                    help="The name of model")

parser.add_argument('--normalize',  nargs='*',
                    help="Defines what normalization method is applied to input images of models. (divide, minmax, standard, mean).\
                    \
                    \ndivide, means that each image is divided by the given value. \
                        \n\tFor example: \"--normalize divide 255\" means that, each image is divided by 255. \
                        for the purpose of normalization. \
                    \
                    \n\nminmax means that image is arranged between minimum and maximum values provided by user. \
                        \nFor example, \"--normalize minmax -1 1\" means that each images are processed such that \
                        the minimum and maximum valus of image would be -1 and 1, respectivelly. \
                    \
                    \n\nstandard means that image standardized with zero mean and standard deviation of 1. \
                        \nIt should be written like this: --normalize standard \"1 2 3\" \"4 5 6\" , for mean and std values, \
                        respectivelly, if mean and std values are given. To calculate mean and std values from training set, \
                        do not provide values. Instead, along with the key \"std\", provide the key \"whole\" for calculation of \
                        both values from training set, or provide the key \"single\" to process each image with its mean and std values.\
                        \n\tFor example: \"--normalize standard whole\" calculates the mean and std values from whole training set.\
                    \
                    \n\nmean, subtracts the mean of image from image. Similar to the std method, if mean value(s for each channel) \
                        is given, each image is processed by subtracting this (those) mean value(s of each channels) from images. \
                        If the mean value(s) is not given, the mean value(s) is calculated from training set, than." )

parser.add_argument('--normalizeback', action='store_true', default=False,
                    help='The produced image by model to be normalized back with relevant values.')

parser.add_argument('--normalizeground', action='store_true', default=False,
                    help='Normalization is applied to ground truth images, if it is given.')

parser.add_argument("--outputdir", type=str, default='',
                    help="Path to output folder")

parser.add_argument('--plotimage', action="store_true", default=False,
                    help ='Used for plotting the output image.')

parser.add_argument("--plotmodel",  action="store_true", default=False,
                    help="Plots model layout.")

parser.add_argument("--predict", action = "store_true", default=False,
                    help='Used to get the prediction scores of the model for given image and with given model weights')

parser.add_argument("--saveimages",  action="store_true", default=False,
                    help="Determines whether images being saved or not while testing. Takes no additional argument. \
                    Used with \"test\" argument. True, if it is given, False, otherwise.")

parser.add_argument("--scale", type=int, default=2,
                    help="The magnification factor.")

parser.add_argument('--seed', type=int, default=19, help='Seed number of number generators to be used for re-producable deterministic models')

parser.add_argument("--shuffle",  action="store_true", default=True,
                    help="Shuffles input files (images) and patches of those files to ensure randomness.")

parser.add_argument('--shutdown', type=str,
                    help='Computer will be shut down after all processes have been done.')

parser.add_argument("--stride", type=int, default=10,
                    help="The stride for generating input images (patches) for training of model.  \
                    It is the same for both directions.")

parser.add_argument('--repeat_test', type=int,
                    help ='Used to do test repeatedly ')

parser.add_argument("--target_cmode", type=str, default="", choices=["YCbCr", "RGB"],
                    help="Traget color mode (RGB or YCbCr) for testing, saving and/or plotting of images. \
                        It will be the same as colormode, if it is not given.")

parser.add_argument("--test",  action="store_true", default=False,
                    help="Used to test models after they have been trained. \
                     Takes no argument with itself. True, if it is given; False, otherwise.")

parser.add_argument("--testpath", type=str, default='',
                    help="Path to the test image file or the directory where test image(s) in")

parser.add_argument("--train", action="store_true", default=False,
                    help="Used to train model. Takes no additional arguments. Training of model be performed, \
                        if it is given. If this option is chosen, instead of 'trainonfly', model trained with \
                        images contained in a .h5 file, which was already prepared. Use 'trainonfly for training \
                         the model on the fly.")

parser.add_argument("--traindir", type=str, default='',
                    help="The directory path where the training images in")

parser.add_argument("--train_with_generator",  action="store_true", default=False,
                    help="Provide this parameter to immediately run training with generator for huge data sizes, after class is built")

parser.add_argument("--trainonfly", action="store_true", default=False,
                     help="Given to traing model with 'on fly' mode of Keras. This method runs in memory. \
                     Instead of taking image patches from an already prepared .h5 file, \
                      it produces training image patches from images in a given folder, and then, \
                      trains the model with these images in memory. Takes no additional arguments")

parser.add_argument("--upscaleimage", default=False,
                    help="Indicates whether model scales the input image by itself or not.\
                     Some models have input image already upscaled with an interpolation method, while others\
                      upscales images from downscaled low resolution images by themselves. If model will upscale\
                       the image, this parameter should be given.")

parser.add_argument("--valdir", type=str, default='',
                    help="The directory path in which the test images is.")

parser.add_argument("--weightpath", type=str, default='',
                    help="Path to the weight file or to the directory where weight file(s) in")

parser.add_argument("--workingdir", type=str, default='',
                    help="Path to working folder")


def getArgs():

    return vars(parser.parse_args())


"""   
if __name__ == '__main__':
    
    args = parser.parse_args()
"""