import matplotlib.pyplot as plt


def make_plot(x, label_plot):
    if label_plot == 'makespan':
        plt.plot(x, label="Deep Q-learning")
        plt.axhline(y=3101, color='r', linestyle='-', label="Q-learning")
        plt.axhline(y=3093, color='g', linestyle='--', label="Best found before")
        plt.xlabel('No episodes')
        plt.ylabel('makespan')
        plt.legend()
        plt.show()
    else:
        plt.plot(x, label=label_plot)
        plt.xlabel('No episodes')
        plt.ylabel(label_plot)
        plt.show()
