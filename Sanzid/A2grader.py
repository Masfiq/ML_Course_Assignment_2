run_my_solution = False

import os
import copy
import signal
import os
import numpy as np
import platform


if run_my_solution:
    from A2solution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    import subprocess, glob, pathlib, shlex

    assignmentNumber = '2'

    nb_name = f'*A{assignmentNumber}solution*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name), None)

    print('\n======================= Code Execution =======================\n')

    print(f'Extracting python code from notebook named {filename} and storing in notebookcode.py')
    if not filename:
        raise Exception(f'Please rename your notebook file to A{assignmentNumber}solution.ipynb'.format(assignmentNumber))

    with open('notebookcode.py', 'w') as outputFile:
        on_windows = platform.system() == "Windows"
        cmd = "where" if on_windows else "which"
        res = subprocess.run([cmd, 'jupyter'], capture_output=True)
        jup = res.stdout[:-1].decode('utf-8')
        comm = f'{jup} nbconvert --to script {nb_name} --stdout --Application.log_level=WARN'
        # print(shlex.split(comm))
        if on_windows:
            subprocess.call(shlex.split(comm), stdout=outputFile, shell=True)
        else:
            subprocess.call(shlex.split(comm), stdout=outputFile)
        
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


exec_grade = 0

def test(points, runthis, correct_str, incorrect_str):
    if (runthis):
        print()
        print('-'*70)
        print(f'----  {points}/{points} points. {correct_str}')
        print('-'*70)
        return points
    else:
        print()
        print('-'*70)
        print(f'----  0/{points} points. {incorrect_str}')
        print('-'*70)
        return 0

# from neuralnetwork import NeuralNetwork

for func in ['NeuralNetwork']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        break
    for method in ['_forward', '_gradients', 'train', 'use']:
        if method not in dir(NeuralNetwork):
            print('CRITICAL ERROR: NeuralNetwork Function named \'{}\' is not defined'.format(method))
            print('  Check the spelling and capitalization of the function name.')
            
def set_weights_for_testing(self):
    for W in self.Ws[:-1]:   # leave output layer weights at zero
        n_weights = W.shape[0] * W.shape[1]
        W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
        for u in range(W.shape[1]):
            W[:, u] += (u - W.shape[1]/2) * 0.2
    # Set output layer weights to zero
    self.Ws[-1][:] = 0
    print('Weights set for testing by calling set_weights_for_testing()')

setattr(NeuralNetwork, 'set_weights_for_testing', set_weights_for_testing)


######################################################################

pts = 10

runthis = '''
n_inputs = 3
n_hiddens = [2, 1]
n_outputs = 2
n_samples = 5

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
T = np.hstack((X, X*2))

nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
nnet.set_weights_for_testing()

# Set standardization variables so use() will run
nnet.X_means = 0
nnet.X_stds = 1
nnet.T_means = 0
nnet.T_stds = 1

Y = nnet.use(X)

Y_correct = np.array([[0., 0.],
    [0., 0.],
    [0., 0.],
    [0., 0.],
    [0., 0.]])
'''

testthis = 'np.allclose(Y, Y_correct, 0.1)'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Y is correct value.',
                       'Y is not equal to Y_correct.')

except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor or use raised the exception.\n')
    print(ex)


######################################################################

pts = 20

runthis = '''
n_inputs = 3
n_hiddens = []   # NO HIDDEN LAYERS.  SO THE NEURAL NET IS JUST A LINEAR MODEL.
n_samples = 5

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
T = np.hstack((X, X*2))
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
nnet.set_weights_for_testing()

nnet.train(X, T, X, T, 1000, 0.01)
Y = nnet.use(X)

Y_correct = np.array([[0.00399238, 0.10399238, 0.20399238, 0.00798476, 0.20798476,
    0.40798476],
    [0.30199619, 0.40199619, 0.50199619, 0.60399238, 0.80399238,
    1.00399238],
    [0.6       , 0.7       , 0.8       , 1.2       , 1.4       ,
    1.6       ],
    [0.89800381, 0.99800381, 1.09800381, 1.79600762, 1.99600762,
    2.19600762],
    [1.19600762, 1.29600762, 1.39600762, 2.39201524, 2.59201524,
    2.79201524]])
'''

testthis = 'np.allclose(Y, Y_correct, 0.5)'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Y is correct value.',
                       'Y is not equal to Y_correct.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor or use raised the exception.\n')
    print(ex)



######################################################################


pts = 20

runthis = '''
n_inputs = 3
n_hiddens = [20, 20, 10, 10, 5]
n_samples = 100

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
T = np.log(X + 0.1)
n_outputs = T.shape[1]

Xtrain = X[np.arange(0, n_samples, 2), :]
Ttrain = T[np.arange(0, n_samples, 2), :]
Xval = X[np.arange(1, n_samples, 2), :]
Tval = T[np.arange(1, n_samples, 2), :]

def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

nnet = NeuralNetwork(n_inputs, n_hiddens, n_outputs)
nnet.set_weights_for_testing()

nnet.train(Xtrain, Ttrain, Xval, Tval, 6000, 0.01)
Yval = nnet.use(Xval)
error = rmse(Yval, Tval)
print(f'RMSE {error:.4f}')
'''

testthis = '0.0 < error < 0.2'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'error is in correct range of 0.0 to 0.2.',
                       'error is not in range 0.0 to 0.2.')
except Exception as ex:
    print('\n--- 0/{} points. NeuralNetwork constructor, train, or use raised the exception\n'.format(pts))
    print(ex)



######################################################################


pts = 20

runthis = '''
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

n_hiddens_each_layer = [5, 10]
n_epochs = 1000
learning_rate = 0.1

nnet, Ytrain, Yvalidate = create_model(X, T, X, T,
                                       n_hiddens_each_layer, n_epochs, learning_rate)

Y_correct = np.array([[0], [1], [1], [0]])
'''

testthis = 'np.allclose(Ytrain, Y_correct, 0.1) and np.allclose(Yvalidate, Y_correct, 0.1)'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Ytrain and Yvalidate have correct values.',
                       'Ytrain and Yvalidate do not have correct values')
except Exception as ex:
    print('\n--- 0/{} points. create_model or NeuralNetwork functions raised the exception\n'.format(pts))
    print(ex)





name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 70')
print('\n REMEMBER, YOUR FINAL EXECUTION GRADE MAY BE DIFFERENT,\n BECAUSE DIFFERENT TESTS WILL BE RUN.')
print('='*70)

print('''
 _ / 6 points. 1. How is the training RMSE curve affected by the number of hidden layers
                  and the number of units in each layer?

 _ / 6 points. 2. How is the final training and validation RMSE affected by the number of epochs
                  and learning rate?

 _ / 6 points. 3. How much do the final training and validation RMSE values vary for different
                  training runs that differ only in the intial random weights?

 _ / 6 points. 4. How well does your best model do in predicting heading and cooling load?
                  In other words, what does an RMSE of a particular value mean in relation
                  to the target values?

 _ / 6 points. 5. What was the hardest part of this assignment?  What is an estimate of the
                  number of hours you spent on this assignment?''')


print()
print('='*70)
print('{} Experiments and Discussion Grade is __ / 30'.format(name))
print('='*70)

print()
print('='*70)
print('{} FINAL GRADE is  ___ / 100'.format(name))
print('='*70)

print('''
Extra Credit (1 point):

Which inputs does your trained neural network find to be most signficant?

There are many ways to answer this. For this extra credit, print the absolute values of the weights in the first hidden layer for all units in that layer. The "all units" is the hard part. Try just taking the mean of the absolute values of the weights for each input across all units. Do the results make sense to you?


Extra Credit (1 point):

Try using a matplotlib.pyplot call like

plt.imshow(np.abs(nnet.Ws[0]), interpolation='nearest')
plt.colorbar()

to see if you can visually see patterns in the weight magnitudes. Describe what you see.
''')

print('\n{} EXTRA CREDIT is 0 / 2'.format(name))

if True and run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

