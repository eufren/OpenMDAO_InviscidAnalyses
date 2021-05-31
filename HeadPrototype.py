from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


Nu = 1.81e-5
Const_vel = 10

def ue(s):
    return Const_vel


def fdUe_ds(s):
    return 0


def re_theta(Ue, theta, nu):
    return Ue*theta/nu


def cf(H, Re_theta):
    return 0.01013/(np.log10(Re_theta)-1.02) - 0.00075


def h(H1):
    if H1 < 3.3:
        return 3
    elif H1 < 5.3:
        return 0.6778 + 1.1536*(H1-3.3)**-0.326
    elif H1 > 5.3:
        return 1.1 + 0.86*(H1-3.3)**-0.777


def h1_and_dh1dh(H):  # Houwink-Veldman, line 4496 in VIQSI2d.f
    if H < 2.732:
        H1 = (0.5 * H + 1) * H / (H - 1)
        dH1dH = ((H**2) - (2*H) - 2) / (2*((H - 1)**2))
    else:
        ht = 0.5 * (H - 2.732) + 2.732
        if ht < 4:
            H1 = (0.5 * ht + 1) * ht / (ht - 1)
            dH1dH = ((ht**2) + (2*ht) - 2) / (4*((ht - 1)**2))
        else:
            H1 = 1.75 + 5.52273 * ht / (ht + 5.818181)
            dH1dH = 0.5 * (5.52273 * 5.818181) / (ht + 5.818181) ** 2
    return H1, dH1dH


def ce(H1):
    return 0.0306/((H1-3)**0.6169)


def f1(Cf, H, theta, Ue, dUe_ds):
    return 0.5*Cf - (H+1)*(theta/Ue)*dUe_ds


def gradients(s, y):
    Ue = ue(s)
    dUe_ds = fdUe_ds(s)
    theta = y[0]/Ue
    H = y[1]
    Re_theta = re_theta(Ue, theta, Nu)
    H1, dH1dH = h1_and_dh1dh(H)
    Cf = cf(H, Re_theta)
    CE = ce(H1)
    F1 = f1(Cf, H, theta, Ue, dUe_ds)

    dtheta_ds = 0.5*Cf - (theta/Ue)*(H+2)*dUe_ds
    dH_ds = (dH1dH**-1)*((CE/theta) - ((H1*dtheta_ds)/theta) - ((H1*dUe_ds)/Ue))
    return np.asarray([dtheta_ds, dH_ds])


def turbsolve(s, theta_0, H_0):
    sol = solve_ivp(gradients, (s[0],s[-1]), [theta_0, H_0], dense_output=True, t_eval=np.linspace(s[0], s[-1], 400))
    return sol


def turbtest():
    sol = solve_ivp(gradients, (0,1), [0.001, 2.4], dense_output=True, t_eval=np.linspace(0, 1, 400), method='Radau')


    H = sol.y[1, :]
    d1 = sol.y[0, :]*H

    H1 = np.array([h1_and_dh1dh(h)[0] for h in H])

    Rex = Const_vel*sol.t/Nu
    prandlt_d1 = 0.37*sol.t*(Rex**-0.2)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(sol.t, sol.y[0,:])
    axs[0, 0].set_title('Momentum Thickness')
    axs[0, 1].plot(sol.t, H1)
    axs[0, 1].set_title('H1')
    axs[1, 0].plot(sol.t, H)
    axs[1, 0].set_title('H')
    axs[1, 1].plot(sol.t, d1, label="Calculation")
    axs[1, 1].plot(sol.t, prandlt_d1, label="Empirical")
    axs[1, 1].set_title('Displacement Thickness')
    axs[1, 1].legend()
    plt.show()


sol = solve_ivp(gradients, (0,1), [0.001, 2.4], dense_output=True, t_eval=np.linspace(0, 1, 400), method='Radau')

H = sol.y[1, :]
d1 = sol.y[0, :]*H

H1 = np.array([h1_and_dh1dh(h)[0] for h in H])

Rex = Const_vel*sol.t/Nu
prandlt_d1 = 0.37*sol.t*(Rex**-0.2)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(sol.t, sol.y[0,:])
axs[0, 0].set_title('Momentum Thickness')
axs[0, 1].plot(sol.t, H1)
axs[0, 1].set_title('H1')
axs[1, 0].plot(sol.t, H)
axs[1, 0].set_title('H')
axs[1, 1].plot(sol.t, d1, label="Calculation")
axs[1, 1].plot(sol.t, prandlt_d1, label="Empirical")
axs[1, 1].set_title('Displacement Thickness')
axs[1, 1].legend()
plt.show()