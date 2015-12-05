#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
##                                         SVM Classification Model Trainer                                           ##
##                                                  Juan Sarria                                                       ##
##                                                December 2, 2015                                                    ##
########################################################################################################################
## Version #: ?
## Description: Code that constructs a Support Vector Classifier using scikit-learn to be able to classify flower
##              images. Images used for training, validation, and testing data are originally credited to the Oxford
##              Visual Geometry group and can be found in the following link:
##                                     http://www.robots.ox.ac.uk/~vgg/data/flowers/
########################################################################################################################

########################################################################################################################
import sys, colorsys, random,numpy, Image, timeit
from glob import glob
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import chi2_kernel, linear_kernel
from sklearn.svm import SVC


NUM_CLASSES = ['17', '102']
IMAGE_TYPES = {'regular' :'/jpg','square' :'/square_images','thumb' :'/thumbnails','tiny' :'/tiny_images_32'}
#----------------------------------------------------------------------------------------------------------------------#

def hsv_image_calculate(rgb_image):
    hsv_image = []
    for pixel in rgb_image:
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        hsv_image.append((h,s,v))
    return hsv_image

#----------------------------------------------------------------------------------------------------------------------#
def kmeans_cluster(numWords, fitted_data):
    kmodel = KMeans(n_clusters = numWords, init= 'k-means++', n_jobs=-1)
    kmodel.fit(fitted_data)
    return kmodel
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------
def main(top_folder, image_folder):

    init_time = timeit.default_timer()
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    # Download labels for pictures
    print('Getting Labels  --------------------------------------------------')
    mat = loadmat(top_folder + '/' + top_folder +'imagelabels.mat')
    labels = mat['labels'][0].tolist()
    print('Done')
    print('------------------------------------------------------------------\n')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Getting Info on Train, Val, Test Data   --------------------------')
    mat = loadmat(top_folder + '/' + top_folder + 'datasplits.mat')
    trn = [mat['trn1'][0].tolist(), mat['trn2'][0].tolist(), mat['trn3'][0].tolist()]
    tst = [mat['tst1'][0].tolist(), mat['tst2'][0].tolist(), mat['tst3'][0].tolist()]
    val = [mat['val1'][0].tolist(), mat['val2'][0].tolist(), mat['val3'][0].tolist()]
    print('Done')
    print('------------------------------------------------------------------\n')


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Getting Images Names  --------------------------------------------')
    imagefiles = []
    # Training model on 32X32 images
    for imagefile in glob(top_folder + image_folder + '/image_*.jpg'):
        imagefiles.append(imagefile)
    imagefiles = sorted(imagefiles)
    print('Done')
    print('------------------------------------------------------------------\n')

    fout_test = open('image_labels.txt','w')
    for ind, element in enumerate(imagefiles):
        fout_test.write(str(labels[ind]) + ' ' + element + '\n')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Getting Images H,S,V  --------------------------------------------')
    print('and Seperating Traning, Validation, and Test Data  ---------------')
    trn_hsv_images = []
    val_hsv_images = []
    tst_hsv_images = []

    trn_labels = []
    val_labels = []
    tst_labels = []

    for idx, imagefile in enumerate(imagefiles):
        img = Image.open(imagefile)
        img.load()

        # Get tuple of r,g,b values for each pixel
        rgb_image = list(img.getdata())

        # transform r,g,b to h,s,v
        hsv_image = hsv_image_calculate(rgb_image)
        '''
        hsv_image = []
        for pixel in rgb_image:
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            hsv_image.append((h,s,v))
        '''

        # sort between training, testing, and validation
        if idx+1 in trn[0]:
            trn_hsv_images.append(hsv_image)
            trn_labels.append(labels[idx])
        elif idx+1 in val[0]:
            val_hsv_images.append(hsv_image)
            val_labels.append(labels[idx])
        elif idx+1 in tst[0]:
            tst_hsv_images.append(hsv_image)
            tst_labels.append(labels[idx])
        else:
            print idx

    print('# of training images: ' + str(len(trn_hsv_images)))
    print('# of validation images: ' + str(len(val_hsv_images)))
    print('# of test images: ' + str(len(tst_hsv_images)))
    print('')
    print('# of traning labels:' + str(len(trn_labels)))
    print('# of validation labels:' + str(len(val_labels)))
    print('# of test labels:' + str(len(val_labels)))
    print('------------------------------------------------------------------\n')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Getting H,S,V Features from Training Set  ------------------------')
    start_time = timeit.default_timer()
    hsv_features = []
    for image in trn_hsv_images:
        # getting features from 1/3 of the training images
        if random.randrange(0,4,1) == 1:
            hsv_features = hsv_features + image
    hsv_features = numpy.asarray(hsv_features)
    print 'Feature matrix: ' + str(hsv_features.shape)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time))
    print('------------------------------------------------------------------\n')




#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Training and Validation  ----------------------------------------')
    training_time = timeit.default_timer()
    minError = -1
    bestWords = -1
    for numWords in range(200, 1001, 100):
        start_time2 = timeit.default_timer()

    ####################################################################################################################
        print(str(numWords) + ' Word Vocabulary  ---------------------------------------------')

        print('Clustering Features...')
        start_time3 = timeit.default_timer()
        kmodel = kmeans_cluster(numWords, hsv_features)
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Hot Encoding Training Data...')
        start_time3 = timeit.default_timer()
        trn_hsv_hot_vectors = []
        for image in trn_hsv_images:
            image_as_clusters = kmodel.predict(numpy.asarray(image)).tolist()
            hot_vector = [0]*numWords
            for  element in image_as_clusters:
                hot_vector[element]+=1
            trn_hsv_hot_vectors.append(hot_vector)
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################

        print('Calculate uf...')
        start_time3 = timeit.default_timer()
        sum_chi_distance = 0
        count = 0
        for i in range(0,len(trn_hsv_hot_vectors),1):
            for j in range(i+1,len(trn_hsv_hot_vectors),1):
                x = numpy.asarray(trn_hsv_hot_vectors[i])
                y = numpy.asarray(trn_hsv_hot_vectors[j])
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    z = numpy.true_divide((x-y)**2,x+y)
                    z[z == numpy.inf] = 0
                    z = numpy.nan_to_num(z)
                    sum_chi_distance += z.sum()
                count += 1

        uf = 2*count/sum_chi_distance
        #uf = 0.5
        print('uf = ' + str(uf))
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Computing Training Kernel...')
        start_time3 = timeit.default_timer()
        #K = linear_kernel(numpy.asarray(trn_hsv_hot_vectors))
        K = chi2_kernel(numpy.asarray(trn_hsv_hot_vectors), gamma=uf)
        print('Size of kernel: ' + str(K.shape))
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Applying Kernel to SVM...')
        start_time3 = timeit.default_timer()
        svm = SVC(kernel='precomputed').fit(K,numpy.asarray(trn_labels))
        trn_predict = svm.predict(K).tolist()
        trn_error = 0
        for i in range(0,len(trn_predict),1):
            if trn_labels[i] != trn_predict[i]:
                trn_error+=1
        print('Error on training data: ' + str(trn_error) + '/' + str(len(trn_labels)))
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Hot Encoding Validation Data...')
        start_time3 = timeit.default_timer()
        val_hsv_hot_vectors = []
        for image in val_hsv_images:
            image_as_clusters = kmodel.predict(numpy.asarray(image)).tolist()
            hot_vector = [0]*numWords
            for  element in image_as_clusters:
                hot_vector[element]+=1
            val_hsv_hot_vectors.append(hot_vector)
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Computing Validation Kernel...')
        start_time3 = timeit.default_timer()
        #K = linear_kernel(X=numpy.asarray(val_hsv_hot_vectors), Y=numpy.asarray(trn_hsv_hot_vectors))
        K = chi2_kernel(X=numpy.asarray(val_hsv_hot_vectors), Y=numpy.asarray(trn_hsv_hot_vectors), gamma=uf)
        print('Size of kernel: ' + str(K.shape))
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Predicting Values for Validation Data...')
        start_time3 = timeit.default_timer()
        val_predict = svm.predict(K)
        print 'Prediction array size: ' + str(val_predict.shape)
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        print('Calculating Error...')
        start_time3 = timeit.default_timer()
        val_predict = val_predict.tolist()
        error = 0
        for i in range(0,len(val_predict),1):
            if val_labels[i] != val_predict[i]:
                error+=1
        print('Error = ' + str(error) + '/' + str(len(val_labels)))
        print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

    ####################################################################################################################
        if minError == -1 or error <= minError:
            minError = error
            bestWords = numWords

        print('------------------------------------------------------------------')
        print('Elapsed Time for ' + str(numWords) + ' words: ' + str(timeit.default_timer() - start_time2))
        print('------------------------------------------------------------------\n')


    print('Training elapsed Time: ' + str(timeit.default_timer() - training_time))
    print('Best number of words is ' + str(bestWords))
    print('Error = ' + str(minError)+ '\n')
    print('------------------------------------------------------------------\n')
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#
    print('Testing  --------------------------------------------------------')
    testing_time = timeit.default_timer()
########################################################################################################################
    print('Getting H,S,V Features from Training and Validation Set...')
    start_time = timeit.default_timer()
    hsv_features = []
    for image in trn_hsv_images + val_hsv_images:
        # getting features from 1/4 of the training and validation images
        if random.randrange(0,4,1) == 1:
            hsv_features = hsv_features + image

    hsv_features = numpy.asarray(hsv_features)
    print 'Feature matrix: ' + str(hsv_features.shape)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Clustering Features...')
    start_time = timeit.default_timer()
    kmodel = kmeans_cluster(bestWords, hsv_features)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Hot Encoding Training Data...')
    start_time = timeit.default_timer()
    trn_hsv_hot_vectors = []
    for image in trn_hsv_images + val_hsv_images:
        image_as_clusters = kmodel.predict(numpy.asarray(image)).tolist()
        hot_vector = [0]*bestWords
        for  element in image_as_clusters:
            hot_vector[element]+=1
        trn_hsv_hot_vectors.append(hot_vector)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################

    print('Calculate uf...')
    start_time = timeit.default_timer()
    sum_chi_distance = 0
    count = 0
    for i in range(0,len(trn_hsv_hot_vectors),1):
        for j in range(i+1,len(trn_hsv_hot_vectors),1):
            x = numpy.asarray(trn_hsv_hot_vectors[i])
            y = numpy.asarray(trn_hsv_hot_vectors[j])
            with numpy.errstate(divide='ignore', invalid='ignore'):
                z = numpy.true_divide((x-y)**2,x+y)
                z[z == numpy.inf] = 0
                z = numpy.nan_to_num(z)
                sum_chi_distance += z.sum()
            count += 1
    uf = count/sum_chi_distance
    print('uf = ' + str(uf))
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Computing Training Kernel...')
    start_time = timeit.default_timer()
    #K = linear_kernel(numpy.asarray(trn_hsv_hot_vectors))
    K = chi2_kernel(numpy.asarray(trn_hsv_hot_vectors), gamma=uf)
    print('Size of kernel: ' + str(K.shape))
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Applying Kernel to SVM...')
    start_time = timeit.default_timer()
    svm = SVC(kernel='precomputed').fit(K,numpy.asarray(trn_labels + val_labels))
    trn_predict = svm.predict(K).tolist()
    trn_error = 0
    trn_labels = trn_labels + val_labels #Overwrites trn_labels
    for i in range(0,len(trn_predict),1):
        if trn_labels[i] != trn_predict[i]:
            trn_error+=1
    print('Error on training data: ' + str(trn_error) + '/' + str(len(trn_labels+val_labels)))
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Hot Encoding Testing Data...')
    start_time = timeit.default_timer()
    tst_hsv_hot_vectors = []
    for image in tst_hsv_images:
        image_as_clusters = kmodel.predict(numpy.asarray(image)).tolist()
        hot_vector = [0]*bestWords
        for  element in image_as_clusters:
            hot_vector[element]+=1
        tst_hsv_hot_vectors.append(hot_vector)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Computing Testing Kernel...')
    start_time = timeit.default_timer()
    #K = linear_kernel(X=numpy.asarray(tst_hsv_hot_vectors), Y=numpy.asarray(trn_hsv_hot_vectors))
    K = chi2_kernel(X=numpy.asarray(tst_hsv_hot_vectors), Y=numpy.asarray(trn_hsv_hot_vectors), gamma=uf)
    print('Size of kernel: ' + str(K.shape))
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Predicting Values for Testing Data...')
    start_time = timeit.default_timer()
    tst_predict = svm.predict(K)
    print 'Prediction array size: ' + str(tst_predict.shape)
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time) + '\n')

########################################################################################################################
    print('Calculating Error...')
    start_time3 = timeit.default_timer()
    tst_predict = tst_predict.tolist()
    error = 0
    for i in range(0,len(tst_predict),1):
        if tst_labels[i] != tst_predict[i]:
            error+=1
    print('Error = ' + str(error) + '/' + str(len(tst_labels)))
    print('Elapsed Time: ' + str(timeit.default_timer() - start_time3) + '\n')

########################################################################################################################
    print('Testing Elapsed Time: ' + str(timeit.default_timer() - testing_time) + '\n')
    full_time = init_time - timeit.default_timer()

    print('Writing to file...')
    fout = open('svm_results.txt','w')
    fout.write('Execution Time: ' + str(full_time) + '\n')
    fout.write('Minimum Validation Error: ' + str(minError) + '/' + str(len(val_labels)) + '\n')
    fout.write('Testing Error: ' + str(error) + '/' + str(len(tst_labels)) + '\n\n')
    fout.write('Real:   Predict:' + '\n')

    for i in range(0,len(tst_labels),1):
        fout.write(str(tst_labels[i]) + '    ' + str(tst_predict[i]))
        if tst_labels[i] == tst_predict[i]:
            fout.write('    X')
        fout.write('\n------------------------------\n')

    fout.close()

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#



if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in NUM_CLASSES and sys.argv[2] in IMAGE_TYPES:
        #folder for either 17 class set and 102 class set
        top_folder = sys.argv[1]
        #images that are being scanned
        image_folder = IMAGE_TYPES[sys.argv[2]]
        main(top_folder, image_folder)
    else:
        print 'Incorrect input parameter'