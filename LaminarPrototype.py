from scipy.integrate import solve_ivp
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


Nu = 1.81e-5
Const_vel = 10


def ue(s):
    return Const_vel


def fdUe_ds(s):
    return 0


def re_theta(s):
    return ue(s)*theta(s)/Nu


def theta(s):
    theta2ue6 = np.array([0.45*Nu*quad(lambda ss: ue(ss)**5, 0, ss)[0] for ss in s])
    Theta = np.sqrt(theta2ue6/(ue(s)**6))
    return Theta


def lambbda(s):
    return theta(s)**2 * (1/Nu) * fdUe_ds(s)


def landH(s):
    Lambdas = lambbda(s)
    def test_lambda(Lambda):
        if Lambda <= -0.1:
            l = 0.22 + 1.402*(-0.1) + (0.018*-0.1)/(-0.1+0.107)
            H = 2.088 + 0.0731/(-0.1+0.14)
        elif -0.1 < Lambda <= 0:
            l = 0.22 + 1.402*Lambda + (0.018*Lambda)/(Lambda+0.107)
            H = 2.088 + 0.0731/(Lambda+0.14)
        elif 0 <= Lambda <= 0.1:
            l = 0.22 + 1.57*Lambda - 1.8*(Lambda**2)
            H = 2.61 - 3.75*Lambda + 5.24*(Lambda**2)
        elif Lambda >= 0.1:
            l = 0.22 + 1.57*0.1 - 1.8*0.01
            H = 2.61 - 3.75*0.1 + 5.24*0.01
        return l, H
    return np.array([test_lambda(l) for l in Lambdas])


def cf(s):
    l = landH(s)[:, 0]
    return 2*l/re_theta(s)


def dstar(s):
    H = landH(s)[:, 1]
    return theta(s)*H

def lamtest():
    s = np.linspace(0, 1, 100)
    blasius_d1 = 1.72*np.sqrt(Nu*s/ue(s))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(s, theta(s))
    axs[0, 0].set_title('Momentum Thickness')
    axs[0, 1].plot(s, cf(s))
    axs[0, 1].set_title('Cf')
    axs[1, 0].plot(s, landH(s)[:, 1])
    axs[1, 0].set_title('H')
    axs[1, 1].plot(s, dstar(s), label="Calculation")
    axs[1, 1].plot(s, blasius_d1, label="Empirical")
    axs[1, 1].set_title('Displacement Thickness')
    axs[1, 1].legend()
    plt.show()