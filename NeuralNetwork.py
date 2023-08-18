import numpy as np

#Functions
def Cost(y_hat, y):
    return (y_hat - y)**2

def L_prime(y_hat, y):
    return 2*(y_hat - y)

def f_prime():
    return 1 - np.tanh(x)**2

#Initialization
n = (2, 3, 2, 1)
x = np.repeat([1,1], n[1]).reshape(n[1],n[0])
y = x[0][0]*x[0][1]
w, b = np.empty((len(n)-1,), dtype=object), np.empty((len(n)-1,), dtype=object)

for i in range(len(n)-1):
    w[i] = np.random.rand(n[i+1], n[i])
    b[i] = np.random.rand(n[i+1])

#w is indexed by w[layer][neuron][weight]
#b is indexed by b[neuron][bias]

#Forward Propagation
z, a = np.empty((len(w),), dtype=object), np.empty((len(w)+1,), dtype=object)
a[0] = x

for i in range(len(n)-1):
    z[i] = np.sum(np.multiply(a[i], w[i]), axis=1) + b[i]
    if i < len(n)-2:
        a[i+1] = np.repeat(np.tanh(z[i]), n[i+2]).reshape(n[i+2], n[i+1])
    else:
        a[i+1] = z[i]

#len(a) = len(z) + 1

#Backward Propagation
w_grad = w.copy()
b_grad = b.copy()

print("w({}): {}\n\nz({}): {}\n\na({}): {}\n\nb({}): {}".format(len(w), w, len(z), z, len(a), a, len(b), b))

#dC/dW[0] = dC/dY * dY/dW[0] = 2*(y_hat-y) * a[len(a)-2][0][0,1]

#dC/dW[-1][0] = [(dC/dW[0])/(dY/dW)] * dY/dA * dA/dF * dF/dZ * dZ/dW[-1][0] = [w_grad[0]/(dY/dW)] * 1 * 1 * f'(z) * a[len(a)-3][0][0]

#dC/dW[-2][0] = [(dC/dW[-1][0])/(dZ/dW[-1][0])] * dZ/dA * dA/dF * dF/dZ * dZ/dW[-2][0] = [w_grad[-1][0]/(dZ/dW[-1][0])] * w[-1][0] * 1 * f'(z) * a[len(a)-4][0][0]

#dC/dW[-3][0] = [(dC/dW[-2][0])/(dZ/dW[-2][0])] * dZ/dA * dA/dF * dF/dZ * dZ/dW[-3][0] = [w_grad[-2][0]/(dZ/dW[-2][0])] * w[-2][0] * 1 * f'(z) * a[len(a)-5][0][0] <--- different neurons only affect last term
#dC/dW[-3][1] = [(dC/dW[-2][0])/(dZ/dW[-2][0])] * dZ/dA * dA/dF * dF/dZ * dZ/dW[-3][1] = [w_grad[-2][0]/(dZ/dW[-2][0])] * w[-2][0] * 1 * f'(z) * a[len(a)-5][0][1] <--- save everything except last term to dW and then compute in w_grad for efficiency

#Weight Backprop Generalized Form
#dC/dW[i][j] = [(dC/dW[i+1][0])/(dZ/dW[i+1][0])] * dZ/dA * dA/dF * dF/dZ * dZ/dW[i][j] = [w_grad[i+1][0]/(dZ/dW[i+1][0])] * w[i+1][0] * 1 * f'(z) * a[len(a)-2-i][0][j]

#if layer is the last layer:
#   w_grad[layer][neuron][weight] = L_prime(y_hat, y) * a[len(a)-2][0][weight]
#else if layer is pentultimate layer:
#   w_grad[layer][neuron][weight] = (w_grad[layer+1][?][?]/a[len(a)-2][0][?]) * f_prime(z[len(z)-2][0][weight]) * a[len(a)-3][0][weight]
#else for all other layers:
#   w_grad[layer][neuron][weight] = (w_grad[layer+1][?][?]/a[len(a)-3][0][?]) * w[layer+1][?][?] * f_prime(z[len(z)-1-?][0][weight]) * a[len(a)-2-?][0][weight]

#y = a*w + b

#Weight Gradient 
