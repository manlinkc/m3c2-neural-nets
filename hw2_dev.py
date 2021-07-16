"""Manlin Chawla 01205586"""
"""M3C 2018 Homework 2"""

#Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import timeit
from m1 import nmodel as nm #assumes that hw2_dev.f90 has been compiled with: f2py3 -c hw2_dev.f90 -m m1
# May also use scipy, scikit-learn, and time modules as needed

def read_data(tsize=60000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 70000 matrix, X
    and the corresponding labels are stored in a 70000 element array, y.
    The final 70000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int)%2 #rescale the image, convert the labels to 0s and 1s (For even and odd integers)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def snm_test(X,y,X_test,y_test,omethod,input=(None)):
    """"Comments for Question 1 Part 3:

    An advantage of using the Python+Fortran approach is speed. Fortran being a
    compiled language runs algorithms/code much faster than the equivalent
    algorithm/code in Python. This is because in Fortran the code is reduced to
    machine instructions, then saved and executed. As Python is an interpreted
    language each line of code is converted to machine instructions and conversion
    increases time. For computationally intensive routines and large datasets
    as were involved in the Single Neuron Model it is better to write the algortihm in
    Fortran and then import the module into Python.

    A disadvantage of the Fortran+Python approach is that Python is used widely as a
    general purpose programming language and is a lot simpler to use. It's much harder
    to debug code in Fortran making it harder to write long algorithms such as the SNM.
    On the other hand Python has a simple interface making it easier to understand
    error messages"""

    """Train single neuron model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed"""

    if input == (None):
        d=10000
    else:
        d=input

    n = X.shape[0]

    fvec = np.random.randn(n+1)  #initial fitting parameters

    #Read data in
    nm.data_init(n,d)
    nm.nm_x=X
    nm.nm_y=y

    #Add code to train SNM and evaluate testing test_error
    if omethod==1:
        fvec_f = minimize(nm.snmodel,fvec,method='L-BFGS-B', args=(d,n), jac=True).x
    elif omethod==2:
        fvec_f = nm.sgd(fvec, n, 0, d, 0.1)

    #Calculate activations
    z=np.dot(fvec_f[0:n],X_test) + fvec_f[n]
    a=1/(1+np.exp(-z))
    #Round activations
    rounda=a.round()

    #Count number of images with correct label
    ncorrect=0
    for i in range(len(y_test)):
        if rounda[i] == y_test[i]:
            ncorrect=ncorrect+1

    test_error = 1-(ncorrect/len(y_test))
    #fvec_f = None #Modify to store final fitting parameters after training
    #test_error = None #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def nnm_test(X,y,X_test,y_test,m,omethod,input=(None)):
    """Train neural network model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of neurons in inner layer
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    #This condition is used in nm_analyze
    if input == (None):
        d=10000
    else:
        d=input

    #Initial fitting parameters
    n = X.shape[0]
    fvec = np.random.randn(m*(n+2)+1)

    # Read data in
    nm.data_init(n, d)
    nm.nm_x = X
    nm.nm_y = y

    # Add code to train NNM and evaluate testing error, test_error
    if omethod == 1:
        fvec_f = minimize(nm.nnmodel, fvec, method='L-BFGS-B', args=(n,m,d), jac=True).x
    elif omethod == 2:
        fvec_f = nm.sgd(fvec, n, m, d, 0.1)

    # Preallocate weights,z values and activations
    w_inner = np.zeros((m,n))
    z_inner = np.zeros(m)
    a_inner = np.zeros(m)
    a_outer = np.zeros(len(y_test))

    #Set weights for inner layer
    for i1 in range(n):
        j1 = (i1)*m
        w_inner[:, i1] = fvec_f[j1:j1+m]

    #Set inner and outer layer bias, weights for outer layer
    b_inner = fvec_f[n*m:n*m+m]
    w_outer = fvec_f[n*m+m:n*m+2*m]
    b_outer = fvec_f[n*m+2*m]

    #Calculate z values and activations for inner/outer layer
    for i2 in range(len(y_test)):
        for i3 in range(m):
            z_inner[i3] = np.dot(w_inner[i3, :], X_test[:, i2])+b_inner[i3]
            a_inner[i3] = 1 / (1 + np.exp(-z_inner[i3]))

        z_outer = np.dot(w_outer, a_inner) + b_outer
        a_outer[i2] = 1 / (1+np.exp(-z_outer))

    #Round activations
    rounda_outer=a_outer.round()

    #Round activations and sum the correct values
    ncorrect=0
    for i in range(len(y_test)):
        if rounda_outer[i] == y_test[i]:
            ncorrect=ncorrect+1

    #Calculate test errors
    test_error = 1-(ncorrect/len(y_test))

    #fvec_f = None #Modify to store final fitting parameters after training
    #test_error = None #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def nm_analyze(X,y,X_test,y_test,figurenum):
    """ Analyze performance of single neuron and neural network models
    on even/odd image classification problem
    Add input variables and modify return statement as needed.
    Should be called from
    name==main section below
    """

    """The function nm_analyze code is split into three sections each generating
    a figure that is used to explore and illustrate the key qualitative trends
    in the performance of the single neuron and neural network model.

    Figure 1
    ---------------------------------------------------------------------------
    This figure is a plot comparing the test error of the single neuron model
    (SNM) and the neural network model (NNM) as the number of training images (d)
    are increased. To generate this plot I have varied the training data size from
    1000 to 10000 images and set the NNM to have 3 neurons in the inner layer.
    To visualize the trends between test error and training data images I have
    fitted a regression line to each set of points.

    For both the SNM and NNM, the linear fit regression lines show that there is
    a negative relationship between the number of training data images provided
    to the network and the test error. As the number of images that
    each network uses for training increases, as expected each network becomes more
    precise at the even/odd classification problem. Looking at just the raw data
    points it is surprising to see that for a training data size d=1000 both SNM
    and NNM models generate a very similar test error around 0.150. However as the
    training data size is increased by increments of 1000 images the NNM model shows
    a decrease in the test error but at a steeper slope meaning the test error has
    been reduced by a larger amount than the SNM has for the same amount of training
    data images.

    There are some fluctuations in test error around the regression line and this may be
    due to the random starting values.The raw data points for the SNM show less
    fluctuations around the regression line, meaning the residuals are smaller
    and it's easier to fit a linear relationship. The raw data points for the NNM
    show more fluctuations around the regression line and at some values of d the
    test error increases compared to the previous test error generated for a smaller
    training data size. However, in general there is a linear decrease in test error
    for both models as the training data images increase. To check whether this
    linear relationship holds, I tested for training data size up to 6000 images
    and the trend clearly holds.

    Figure 2
    ----------------------------------------------------------------------------
    This figure is a plot comparing the run time of the single neuron model (SNM)
    and the neural network model (NNM) as the number of training images (d) are
    increased. To generate this plot I have varied the training data size from 1000
    to 10000 images and set the NNM to have 2 neurons in the inner layer. To
    visualize the trends between the run time and training data images I have fitted
    a regression lines to each set of points.

    The plots show that for both the SNM and NNM, the linear fit regression line
    shows that there is a positive relationship between the number of training data
    images provided to the network and the run time. As the number of images that
    each network uses for training increases, as expected each network takes more
    time to run the classification problem.

    Looking at just the raw data points it is surprising to see that for a training
    data size d=1000 both SNM and NNM models have a similar run time around 3-6
    seconds. However as the training data size increases, the runtime for the NNM
    increases at a faster rate than the runtime for the SNM increases. In general
    this means the SNM algorithm is faster but this could be due to more calculations
    being involved in the NNM algorithm. As show in the figure 1 although SNM model
    outdoes the NNM model in run time, the NNM model outdoes the SNM model in
    precision. So there is a payoff between performance and speed.

    Figure 3
    -----------------------------------------------------------------------------
    This figure is a plot of the test errors as the number of neurons are increased
    in the neural network model (NNM). To generate this plot I have number of neurons
    in the inner layer from 1 to 6.The red points mark the test error and the red
    line is a fitted regression line to each set of points.

    The linear fit regression line shows that there is a negative relationship
    between the test error and the number of neurons in the NNM model. As the
    number of neurons increase in the inner layer the network becomes more precise.
    To check whether this linear relationship holds, in theory I would test for
    even larger number of neurons in the inner layer. This would require more time
    for such a simulation to run.

    """
    assert figurenum <= 4, "Choose from figures 1,2,3"
    # Using SGD omethod
    omethod=2

    # Figure 1
    # Test error for SNM vs NNM (m=3) as the training data is varied
    if figurenum == 1:
        #Preallocate for efficiency
        snm_testerror=np.zeros(10)
        nnm_testerror=np.zeros(10)

        #Set number of neurons in inner layer of NN model
        m=3
        #Generate test errors
        trainingdata_size=np.linspace(1000,10000,10)

        for i, input in enumerate(trainingdata_size):
            snm_testerror[i]=snm_test(X,y,X_test,y_test,omethod,input)[1]
            nnm_testerror[i]=nnm_test(X,y,X_test,y_test,m,omethod,input)[1]

        #Generate linear fit lines
        slope1, intercept1, r_value, p_value, std_err = stats.linregress(trainingdata_size, snm_testerror)
        line1=slope1*trainingdata_size+intercept1
        slope2, intercept2, r_value, p_value, std_err = stats.linregress(trainingdata_size, nnm_testerror)
        line2=slope2*trainingdata_size+intercept2

        plt.hold(True)
        plt.plot(trainingdata_size,snm_testerror,'.',trainingdata_size,line1, c='b', label='SNM model:Testing Error')
        plt.plot(trainingdata_size,nnm_testerror,'.',trainingdata_size,line2, c='r', label='NNM model:Testing Error')
        plt.xlabel('Training Data Size (d)')
        plt.ylabel('Test Error')
        plt.title('Manlin Chawla: analyze(1) \n Test Error as the size of the training data varies')
        plt.legend()
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([0,0.20])
        plt.hold(False)


    # Figure 2
    # Runtime for SNM vs NNM as the training data is varied
    if figurenum == 2:
        #Preallocate for efficiency
        #Change back to 10 zeros
        snm_runtime=np.zeros(10)
        nnm_runtime=np.zeros(10)

        #Set number of neurons (m=2) in inner layer of NN model
        m=2
        #Generate run times
        #change last value of linspace back to 10
        trainingdata_size=np.linspace(1000,10000,10)

        for i, input in enumerate(trainingdata_size):
            snm_runtime[i]=timeit.timeit(lambda:snm_test(X,y,X_test,y_test,omethod,input),number=1)
            nnm_runtime[i]=timeit.timeit(lambda:nnm_test(X,y,X_test,y_test,m,omethod,input),number=1)

        #Lines of linear fit
        slope1, intercept1, r_value, p_value, std_err = stats.linregress(trainingdata_size, snm_runtime)
        line1=slope1*trainingdata_size+intercept1
        slope2, intercept2, r_value, p_value, std_err = stats.linregress(trainingdata_size, nnm_runtime)
        line2=slope2*trainingdata_size+intercept2

        #Plots
        plt.hold(True)
        plt.plot(trainingdata_size,snm_runtime,'.',trainingdata_size,line1, c='b', label='SNM model:Runtime')
        plt.plot(trainingdata_size,nnm_runtime,'.',trainingdata_size,line2, c='r', label='NNM model:Runtime')
        plt.xlabel('Training Data Size (d)')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: analyze(2) \n Runtime as the size of the training data varies')
        plt.legend()
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        #axes.set_ylim([0,0.20])
        plt.hold(False)


    # Figure 3
    # Test error for NNM as the number of neurons in inner layer is svaried
    if figurenum == 3:
        #Preallocate for efficiency
        #Change zeros back to 6
        nnm_testerror=np.zeros(6)

        #Set number of neurons in inner layer of NN model
        #change last place of linspace back to 6
        numofneurons=np.linspace(1,6,6,dtype=int)

        for i, m in enumerate(numofneurons):
            nnm_testerror[i]=nnm_test(X,y,X_test,y_test,m,omethod,input=(None))[1]

        #Lines of linear fit
        slope1, intercept1, r_value, p_value, std_err = stats.linregress(numofneurons, nnm_testerror)
        line1=slope1*numofneurons+intercept1

        plt.hold(True)
        plt.plot(numofneurons,nnm_testerror,'.',numofneurons,line1, c='r', label='NNM model:Testing Error')
        plt.xlabel('Number of neurons in inner layer (m)')
        plt.ylabel('Test Error')
        plt.title('Manlin Chawla: analyze(3) \n Test Error as the number of neurons in inner layer of NN model are varied')
        plt.legend()
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([0,0.20])
        plt.hold(False)


    return None

#--------------------------------------------
def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None

#--------------------------------------------
if _name_ == '_main_':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    #Read data
    [X,y,X_test,y_test]=read_data()

    output = nm_analyze(X,y,X_test,y_test,1)
    plt.savefig('manlin1.png', bbox_inches="tight")
    plt.show()
    plt.clf()

    # Generates figure 2, formats title and axis label
    output = nm_analyze(X,y,X_test,y_test,2)
    plt.savefig('manlin2.png', bbox_inches="tight")
    plt.show()
    plt.clf()

    # Generates figure 3, formats title and axis labels
    output = nm_analyze(X,y,X_test,y_test,3)
    plt.savefig('manlin3.png', bbox_inches="tight")
    plt.show()
    plt.clf()
#--------------------------------------------
