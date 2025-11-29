import matplotlib.pyplot as plt

def plot_temperature(x, T):
    plt.plot(x, T)
    plt.xlim(min(x), max(x))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.show()