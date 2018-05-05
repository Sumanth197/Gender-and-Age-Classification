# Gender-and-Age-Classification

VGG face network weights can be found here ----> (https://pan.baidu.com/s/1F3d1pXROnvTjebSI4fxUmw).

Dataset can be found here [Adience Benchmark](https://www.openu.ac.il/home/hassner/Adience/data.html#agegender). About 29k images, 2 categories in gender, 8 categories in age.

Using pre-trained conv layers of VGG face network and fine-tuning fully connected layers gets **91%** accuracy in gender and **55%** in age because of Using selu as the activation function can get **2% improvement** in gender estimation.

Actually Model_origin get **85.9%** accuracy in gender and **49.5%** in age.

Model_origin can be found here ----> Age and Gender Classification using Convolutional Neural Networks.
