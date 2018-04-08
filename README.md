# JosiahNet: live coding a GOLANG Deep Learning Library

Inspired by [Joel Grus's youtube live coding](https://github.com/joelgrus/joelnet) of a deep learning frameworks, I wondered as a very new to GOLANG user could I live code a deep learning framework?

It ended up taking a few hours to code this up as I didn't have numpy as a starting point, but I found GOLANG to be very suitable for implementing a **Deep Learning Library in Go with no dependencies**.

Creating this way a great way for me to learn more about Go package creation and non-trivial uses of interfaces.

### Things I might add to this library in the future:
- Data Parallel training with Go Channels
- MNIST example

### Things this library is:
- a simple self contained way to learn about deep learning
- a way to learn about how matix algebra can be implemented from scratch
- a fun toy example

### Things this is not:
- a production deep learning lib for Go, see: [Gorgonia](https://github.com/gorgonia/gorgonia)

To play with this yourself: `go get github.com/thenomemac/josiahnet/jnet`

Run the XOR example:

```bash
 2018-04-07 21:30:50 ⌚  thenome-lpc-13 in ~/gocode/src/github.com/thenomemac/josiahnet
○ → go run examples/xor.go 
----- Begin Training -----

Epoch/Loss: 0	| 87.865
Epoch/Loss: 10	| 1.928
Epoch/Loss: 20	| 1.261
Epoch/Loss: 30	| 0.906
Epoch/Loss: 40	| 0.667
Epoch/Loss: 50	| 0.492
Epoch/Loss: 60	| 0.365
Epoch/Loss: 70	| 0.281
Epoch/Loss: 80	| 0.220
Epoch/Loss: 90	| 0.170
Epoch/Loss: 100	| 0.132
Epoch/Loss: 110	| 0.103
Epoch/Loss: 120	| 0.080
Epoch/Loss: 130	| 0.062
Epoch/Loss: 140	| 0.048
Epoch/Loss: 150	| 0.038
Epoch/Loss: 160	| 0.030
Epoch/Loss: 170	| 0.023
Epoch/Loss: 180	| 0.018
Epoch/Loss: 190	| 0.015
Epoch/Loss: 200	| 0.012
Epoch/Loss: 210	| 0.009
Epoch/Loss: 220	| 0.008
Epoch/Loss: 230	| 0.006
Epoch/Loss: 240	| 0.005
Epoch/Loss: 250	| 0.004
Epoch/Loss: 260	| 0.003
Epoch/Loss: 270	| 0.003
Epoch/Loss: 280	| 0.002
Epoch/Loss: 290	| 0.002
Epoch/Loss: 300	| 0.001
Epoch/Loss: 310	| 0.001
Epoch/Loss: 320	| 0.001
Epoch/Loss: 330	| 0.001
Epoch/Loss: 340	| 0.001
Epoch/Loss: 350	| 0.001
Epoch/Loss: 360	| 0.000
Epoch/Loss: 370	| 0.000
Epoch/Loss: 380	| 0.000
Epoch/Loss: 390	| 0.000
Epoch/Loss: 400	| 0.000
Epoch/Loss: 410	| 0.000
Epoch/Loss: 420	| 0.000
Epoch/Loss: 430	| 0.000
Epoch/Loss: 440	| 0.000
Epoch/Loss: 450	| 0.000
Epoch/Loss: 460	| 0.000
Epoch/Loss: 470	| 0.000
Epoch/Loss: 480	| 0.000
Epoch/Loss: 490	| 0.000

----- End Training -----

Predictions:	 [0.9979224722394722 0.0026328332573238855 0.002664968993301098 0.9972001895643201]
Targets:	 [1 0 0 1]

And we're done! Deep Learning is fun.
```

### FYI: Here's the plan of attack I followed while live coding this library:
1. Tensors
1. Loss Functions
1. Layers
1. Neural Nets
1. Optimizers
1. Data : ended up skipping this
1. Training
1. XOR Example
