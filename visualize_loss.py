import sys
import matplotlib.pyplot as pyplot


def init():
    stats_file = sys.argv[1]

    step = []
    discriminator_loss = []
    generator_loss = []
    with open(stats_file, 'r') as stats_handle:
        for line in stats_handle.readlines():
            epoch, batch, d_loss, g_loss = line.split(',')
            step.append((int(epoch) + 1) * int(batch))
            discriminator_loss.append(float(d_loss))
            generator_loss.append(float(g_loss))

    pyplot.plot(step, discriminator_loss, color='orange')
    pyplot.show()
    pyplot.plot(step, generator_loss, color='green')
    pyplot.show()


if __name__ == "__main__":
    init()
