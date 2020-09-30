# TensorFlow

Tensorflow and NUMBA tests, training MNIST and Fashion MNIST. The second one - example by Francois Chollet, little additions: added CUDA/NUMBA, a fix for possible problem with CUDA installation under Windows:

For GPU/CUDA use download CUDA library and cuDNN 
cuDNN installation is just an extraction of the library files
 If you get an error such as "Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found:
 In my and other cases including the path in the %PATH% Environment var didn't help (Windows).
 The solution that worked was to copy the DLL to the /bin folder of the main CUDA installation, such as:
 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin
