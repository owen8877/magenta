import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

forwards = (True, False)
ks = (16, 32, 64, 128)
hidden_layer_ns = (1, 2)
EPOCH = 500

data = dict()

for forward in forwards:
  data[forward] = dict()
  for k in ks:
    data[forward][k] = dict()
    for hidden_layer_n in hidden_layer_ns:
      loss_file = 'data/loss_k_{:d}_hln_{:d}_dir_{:s}.npz'.format(k, hidden_layer_n, 'forward' if forward else 'backward')
      data[forward][k][hidden_layer_n] = np.load(loss_file)

for forward in forwards:
  plt.figure(1 if forward else 2)
  fig, axs = plt.subplots(1, 2)
  for i, hln in enumerate(hidden_layer_ns):
    ax = axs[i]
    for j, k in enumerate(ks):
      ax.plot(range(EPOCH), data[forward][k][hln]["train_loss"], 'C{:d}'.format(j), label='Train k={:d}'.format(k))
      ax.plot(range(EPOCH), data[forward][k][hln]["test_loss"], 'C{:d}'.format(j), label='Test k={:d}'.format(k), ls=':')
    ax.legend()
    ax.set_title('Direction: {:s} -> {:s}\n{:d} hidden layer(s)'.format('up' if forward else 'bottom', 'bottom' if forward else 'up', hln))
    ax.set_xlabel('epoch')
  axs[0].set_ylabel('mse loss')
  # fig.show()
  fig.savefig('plot/{:s}.eps'.format('forward' if forward else 'backward'), dpi=1000)