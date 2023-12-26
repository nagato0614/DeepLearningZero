import pandas as pd
import matplotlib.pyplot as plt


def main():
    filename = '../debug/step_function.csv'

    df = pd.read_csv(filename)
    df.plot(x='x', y='y', kind='line')
    plt.show()

    filename = '../debug/sigmoid.csv'

    df = pd.read_csv(filename)
    df.plot(x='x', y='y', kind='line')
    plt.show()

    filename = '../debug/relu.csv'

    df = pd.read_csv(filename)
    df.plot(x='x', y='y', kind='line')
    plt.show()

if __name__ == '__main__':
    main()
