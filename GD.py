import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    x1, x2 = x
    
    return 2*x1**2 + 5*x2**2 - 2*x1*x2 + 4

def df1(x):
    x1, x2 = x
    return np.array([4*x1 - 2*x2, 10*x2 - 2*x1])

def f2(x):
    x1, x2 = x
    return x1**4 + 5*x2**2 + 10*x1 + 10*x2 + 10

def df2(x):
    x1, x2 = x
    return np.array([4*x1**3 + 10, 10*x2 + 10])

def f3(x):
    x1, x2 = x
    return -x1**3 + 5*x2**2 + 10*x1 + 10*x2 + 10

def df3(x):
    x1, x2 = x
    return np.array([-3*x1**2 + 10, 10*x2 + 10])

def SGD(f, df, x0, lr=0.01, max_iter=1000, tol=1e-6):
    x = x0.copy()
    hist = [x.copy()]
    obj_vals = [f(x)]
    for _ in range(max_iter):
        grad = df(x)
        x -= lr * grad
        hist.append(x.copy())
        obj_vals.append(f(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, hist, obj_vals

def Momentum(f, df, x0, lr=0.01, gamma=0.9, max_iter=1000, tol=1e-6):
    x = x0.copy()
    v = np.zeros_like(x0)
    hist = [x.copy()]
    obj_vals = [f(x)]
    for _ in range(max_iter):
        grad = df(x)
        v = gamma * v + lr * grad
        x -= v
        hist.append(x.copy())
        obj_vals.append(f(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, hist, obj_vals

def RMSprop(f, df, x0, lr=0.01, beta=0.9, eps=1e-8, max_iter=1000, tol=1e-6):
    x = x0.copy()
    s = np.zeros_like(x0)
    hist = [x.copy()]
    obj_vals = [f(x)]
    for _ in range(max_iter):
        grad = df(x)
        s = beta * s + (1 - beta) * grad**2
        x -= lr * grad / (np.sqrt(s) + eps)
        hist.append(x.copy())
        obj_vals.append(f(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, hist, obj_vals

def Adam(f, df, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, tol=1e-6):
    x = x0.copy()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    hist = [x.copy()]
    obj_vals = [f(x)]
    t = 0
    for _ in range(max_iter):
        t += 1
        grad = df(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(x.copy())
        obj_vals.append(f(x))
        if np.linalg.norm(grad) < tol:
            break
    return x, hist, obj_vals



def GD(F, Op, Nmax, Tol, Lambda, x0):
    if Op == "SGD":
        return SGD(F, hm[F], x0, lr=Lambda, max_iter=Nmax, tol=Tol)
    elif Op == "Momentum":
        return Momentum(F, hm[F], x0, lr=Lambda, max_iter=Nmax, tol=Tol)
    elif Op == "RMSprop":
        return RMSprop(F, hm[F], x0, lr=Lambda, max_iter=Nmax, tol=Tol)
    elif Op == "Adam":
        return Adam(F, hm[F], x0, lr=Lambda, max_iter=Nmax, tol=Tol)
    else:
        print("Invalid Optimizer name")
        exit()

def plot_contour(f, x_min, x_max, y_min, y_max, hist):

    print(hist[-1])
    X = [(i[0]) for i in hist]
    Y = [(i[1]) for i in hist]
    X, Y = np.meshgrid(X, Y)

    Z = f([X, Y])

    plt.figure(figsize=(8, 6))
    contours = plt.contour(X, Y, Z)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.colorbar(label='Objective function value')
    print(functions[f])
    plt.title('Contour Plot of '+functions[f])
    plt.show()

def plot_optimization(obj_vals):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(obj_vals)), obj_vals)
    plt.xlabel("Number of steps")
    plt.ylabel("Objective function value")
    plt.show()

def helper(parameters):
    print('-----------------------------------')
    x0 = np.array(parameters[0])
    x_min = GD(parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],x0)
    print("Minimum Objective function value:", str(parameters[3](x_min[0])))
    print("Local minima for f1:", x_min[0])
    print("Number of steps:", len(x_min[1]))
    plot_contour(parameters[3], parameters[1][0], parameters[1][1], parameters[2][0], parameters[2][1], x_min[1])
    plot_optimization(x_min[2])
    print('-----------------------------------')

if __name__ == '__main__':
    
    hm = {
        f1 : df1,
        f2 : df2,
        f3 : df3
    }

    functions = {
        f1 : '2X1 ^ 2 + 5X2 ^ 2 - 2X1X2 + 4',
        f2 : 'X1^4 +5X2^2+ 10X1 + 10X2 +10',
        f3 : '-X1^3 + 5X2^2+ 10X1 +10X2 +10'
    }

    parameters = [
    [[-0.4,0.4],[-2,2],[-2,2],f1,"Adam",1000,1e-6,0.01],
    [[-0.4,0.4],[-2,2],[-2,2],f1,"Momentum",1000,1e-6,0.01],
    [[-0.4,0.4],[-2,2],[-2,2],f1,"RMSprop",1000,1e-6,0.01],
    [[-0.4,0.4],[-2,2],[-2,2],f1,"SGD",1000,1e-6,0.01],
    [[0.5,0],[-2,2],[-2,2],f2,"Adam",1000,1e-6,0.01],
    [[0.5,0],[-2,2],[-2,2],f2,"Momentum",1000,1e-6,0.01],
    [[0.5,0],[-2,2],[-2,2],f2,"RMSprop",1000,1e-6,0.01],
    [[0.5,0],[-2,2],[-2,2],f2,"SGD",1000,1e-6,0.01],
    [[-0.5,0.5],[-2,2],[-2,2],f3,"Adam",1000,1e-6,0.01],
    [[-0.5,0.5],[-2,2],[-2,2],f3,"Momentum",1000,1e-6,0.01],
    [[-0.5,0.5],[-2,2],[-2,2],f3,"RMSprop",1000,1e-6,0.01],
    [[-0.5,0.5],[-2,2],[-2,2],f3,"SGD",1000,1e-6,0.01]
    ]

    for param in parameters:
        helper(param)


