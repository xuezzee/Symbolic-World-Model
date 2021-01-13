import math
import numpy as np
for i in range(2):
    np.random.seed(9)
    B = 1000000000/5
    pe = [0.1 for _ in range(5)]
    # pe[-1] = 0.9
    pc = [0.5 for _ in range(5)]
    if i == 1:
        pe = [0.1, 0.1, 0.5, 0.6, 0.9]
    He = abs(1 / np.sqrt(2) * (np.random.randn(5)
                                    + 1j * np.random.randn(5)))  # 边缘
    Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(5)
                                          + 1j * np.random.randn(5)))  # 云
    print('H:', He, Hc)
    var_noise = 10**(-2)
    pe = np.array(pe)
    pc = np.array(pc)
    He = np.array(He)
    Hc = np.array(Hc)
    rate_edge = np.zeros((5))
    rate_cloud = np.zeros((5))
    # 边缘网络
    print("!!!!!", pe, pc)
    Ie = 0;
    Ic = 0
    for n in range(5):
        Ie += He[n] ** 2 * pe[n]
        Ic += Hc[n] ** 2 * pc[n]

    for n in range(5):
        print((var_noise + Ie - He[n] ** 2 * pe[n]), He[n] ** 2 * pe[n])
        print("///////", He[n] ** 2 * pe[n] / (var_noise + Ie - He[n] ** 2 * pe[n]))
        rate_edge[n] = np.math.log2(1 + He[n] ** 2 * pe[n] / (var_noise + Ie - He[n] ** 2 * pe[n]))
        rate_cloud[n] = np.math.log2(1 + Hc[n] ** 2 * pc[n] / (var_noise + Ic - Hc[n] ** 2 * pc[n]))

    print(rate_edge, rate_cloud)