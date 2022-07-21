import sys

def init():
    log_file = sys.argv[1]
    stats_file = sys.argv[2]

    stats = []
    with open(log_file, 'r') as log_handle:
        for line in log_handle.readlines():
            epoch = int(line.split('[Epoch ')[1].split('/')[0])
            batch = int(line.split('[Batch ')[1].split('/')[0])
            stats.append({
                'epoch': epoch,
                'batch': batch,
                'discriminator_loss': float(line.split('[D loss: ')[1].split(']')[0]),
                'generator_loss': float(line.split('[G loss: ')[1].split(']')[0]),
            })

    with open(stats_file, 'w') as stats_handle:
        for stat in stats:
            stats_handle.write(
                f'{stat["epoch"]},{stat["batch"]},{stat["discriminator_loss"]},{stat["generator_loss"]}\n')
        stats_handle.close()


if __name__ == "__main__":
    init()
