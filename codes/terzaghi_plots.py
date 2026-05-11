import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import mpmath
import pandas as pd
import h5py
import sympy
from sympy.integrals.transforms import inverse_laplace_transform



# Physical properties
# rho_s = 2500  # kg / m**3
# rho_f = 1000  # kg / m**3
# mu_f = 0.001  # Pa*s
# # mu_f=1e10
# G = 30.0e9  # Pa
# # K_sg = 10.0e9  # Pa
# K_fl = 80.0e9  # Pa
# K_d = 40.0e9  # Pa
# # K_u = 2.6941176470588233 # Pa
# alpha = 0.6  # -
# phi = 0.1  # -
# # M = 4.705882352941176# Pa
# k = 1.5e-13  # m**2
# mu_s = 1.0e10 # Pa*s
rho_s = 2500  # kg / m**3
rho_f = 1000  # kg / m**3
mu_f = 1.0e-3  # Pa*s
G = 30.0e+9  # Pa
# K_sg = 100.0e+9  # Pa
K_fl = 80.0e+9  # Pa
K_d = 40.0e+9  # Pa

alpha = 0.6
phi = 0.1
# alpha = 1e-5
# phi = 0.0

k = 1.5e-13  # m**2

mu_s = 1.0e15 # Pa*s

ymax = 10.0e3  # m
ymin = 0.0  # m
xmax = 10.0  # m
xmin = 0.0  # m
P_0 = -10.0e3  # Pa

# Height of column, m
L = ymax - ymin
H = xmax - xmin

# iterations for series functions
# ITERATIONS = 16000
ITERATIONS = 1600


K_sg = K_d/(1.0 - alpha)
# print(K_sg/1E9)
M = 1.0 / (phi / K_fl + (alpha - phi) / K_sg)  # Pa
K_u = K_d + alpha * alpha * M  # Pa,      Cheng (B.5)
nu = (3.0 * K_d - 2.0 * G) / (2.0 * (3.0 * K_d + G))  # -,       Cheng (B.8)
nu_u = (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G))  # -,       Cheng (B.9)
eta = (3.0 * alpha * G) / (3.0 * K_d + 4.0 * G)  # -,       Cheng (B.11)
S = (3.0 * K_u + 4.0 * G) / (M * (3.0 * K_d + 4.0 * G))  # Pa^{-1}, Cheng (B.14)
c = (k / mu_f) / S  # m^2 / s, Cheng (B.16)

def pressure(z, t):

    # z_star = z / L
    # t_star = (c * t) / (4. * L**2)
    z_star = z
    t_star = t
    return -((P_0 * eta) / (G * S)) * F1(z_star, t_star)

def pressure_PVE(z, t):

    # function for Laplace transformed pressure
    def transformed_p(s):

        G_bar = G*s/(s+(G/mu_s))
        K_bar = K_d#K_d*s/(s+(G/mu_s))
        alpha_bar = alpha#alpha*s/(s+(G/mu_s))

        eta_bar = (3.0 * alpha_bar * G_bar) / (3.0 * K_bar + 4.0 * G_bar)
        inverse_M_bar = (alpha_bar-phi)/K_sg + phi/K_fl
        S_bar = inverse_M_bar + 3*alpha_bar**2/(3*K_bar+4*G_bar)

        # def reduced_F1(ti):
        #     return F1_mpmath(z, ti)

        # F1_bar = laplace_transform(reduced_F1, s)

        return -(P_0*eta_bar/(G_bar*S_bar))*F1_LT(z, s)
    
    return inverse_laplace_transform(transformed_p, t)

def write_pressure_PVE_file(filename):

    # t_list = [1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    z_list = np.arange(0, L+500, 500)/L

    f = open(filename, "w")
    f.write("t*,z*,p\n")

    for t in t_list:
        for z in z_list:
            print("t*="+str(t)+", z*="+str(z))

            p = pressure_PVE(z, t)

            f.write(str(t)+","+str(z)+","+str(p)+"\n")

    f.close()

def plot_pressure_PVE(filename, plot_pylith_PVE=False):

    
    df = pd.read_csv(filename, header=0, delimiter=',')
    # t_list = [1.4e-6, 1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    # color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    # t_list = [1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    # t_list = df.loc[df['t*']].values
    color_list = ["tab:blue", "gold", "limegreen", "lightcoral"]
    color_list_pylith = ["darkblue", "orangered", "darkgreen", "darkred"]
    # print(0.01*((4. * L**2)/c))

    for t_star, cl, clp in zip(t_list, color_list, color_list_pylith):


        df_subset = df.loc[np.abs(df['t*'] - t_star)<1e-3]

        z = df_subset['z*'].values
        p = df_subset['p'].values
        
        plt.plot(p * ((G*S)/(-P_0*eta)), z, label="t*="+str(round(t_star, 3))+", PVE", lw=5, c=cl, alpha=0.8) 
        plt.plot(pressure(z, t_star) * ((G*S)/(-P_0*eta)), z, label="t*="+str(round(t_star, 3))+", PE", lw=7, c=cl, linestyle='dashed', alpha=0.5)  

        if plot_pylith_PVE:

            absolute_t = t_star*((4. * L**2)/c)
            print(t_star, absolute_t)
            y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15.h5")
            plt.plot(p_pylith * ((G*S)/(-P_0*eta)), 1.0-y_pylith/L, label="t*="+str(round(t_star, 3))+", PyLith PVE", lw=5, c=clp, linestyle='solid', marker='o', alpha=0.8, markersize=8, mec='black', markevery=20)      

            # y_pylith_PE, p_pylith_PE, uy_pylith, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain.h5")
            # plt.plot(p_pylith_PE * ((G*S)/(-P_0*eta)), 1.0-y_pylith_PE/L, label="t*="+str(round(t_star, 3))+", PyLith PE", lw=7, c=clp, linestyle='dashed', alpha=0.5, marker='o', markersize=8, mec='black', markevery=20)

        

    plt.xlabel(r"$p / \frac{P_0\eta}{GS}$", fontsize=30)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig("terzaghi_p_PVE_legend.png",  bbox_inches="tight")
    plt.show()

def check_PE_pressure(t):

    z_PE = np.arange(0, L+0.05, 0.05)/L
    t_star = t*c/(4.*L**2)
    print(t_star, t)

    df = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/pressure_PVE_eta_s=1E15_select_times.csv", header=0, delimiter=',')

    df_subset = df.loc[df['t*'] > 0.03]

    z_PVE = df_subset['z*'].values
    p_PVE = df_subset['p'].values

    plt.plot(pressure(z_PE, t_star) * ((G*S)/(-P_0*eta)), z_PE, label="t*="+str(t_star)+", Analytic PE", lw=3, c='blue')
    # plt.plot(p_PVE * ((G*S)/(-P_0*eta)), z_PVE, label="t*="+str(t_star)+", Analytic PVE", lw=3, c="green")         


    y_pylith_PE, p_pylith_PE, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain.h5")
    plt.plot(p_pylith_PE * ((G*S)/(-P_0*eta)), 1.0-y_pylith_PE/L, label="t*="+str(t_star)+", PyLith PE", lw=3, c='red', linestyle='dashed')

    y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E25.h5")
    plt.plot(p_pylith * ((G*S)/(-P_0*eta)), 1.0-y_pylith/L, label="t*="+str(t_star)+", PyLith PVE", lw=3, c='black', linestyle='dotted')

    plt.xlabel(r"$p / \frac{P_0\eta}{GS}$", fontsize=30)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    # plt.savefig("terzaghi_p_PVE_legend.png",  bbox_inches="tight")
    plt.show()


def displacement(z, t):

    # z_star = z / L
    # t_star = (c * t) / (4. * L**2)
    z_star = z
    t_star = t
    return (((P_0 * L * (1.0 - 2.0 * nu_u)) / (2.0 * G * (1.0 - nu_u))) * (1.0 - z_star) + 
            ((P_0 * L * (nu_u - nu)) / (2.0 * G * (1.0 - nu_u) * (1.0 - nu))) * F2(z_star, t_star))


def displacement_PVE(z, t, mu_s_input):

    z_star = z
    t_star = t

    def transformed_u(s):

        G_bar = (G*s)/(s+(G/mu_s_input))
        K_bar = K_d#K_d*s/(s+(G/mu_s))
        alpha_bar = alpha #alpha*s/(s+(G/mu_s))
        # K_sg_bar = K_bar/(1.0-alpha_bar)

        # inverse_M_bar = (alpha_bar-phi)/K_sg_bar + phi/K_fl

        # Ku_bar = K_bar + alpha_bar**2*(1./inverse_M_bar)
        Ku_bar = K_bar + (K_fl*(K_sg-K_bar)**2)/(K_fl*(K_sg - K_bar) + phi*K_sg*(K_sg - K_fl))

        nu_bar = (3*K_bar-2*G_bar)/(2*(3*K_bar+G_bar))
        nu_u_bar = (3*Ku_bar-2*G_bar)/(2*(3*Ku_bar+G_bar))
        # print(nu_u_bar, G_bar)
        return ((P_0*L*(1-2*nu_u_bar))/(2.*s*G_bar*(1-nu_u_bar)))*(1-z_star) + ((P_0*L*(nu_u_bar-nu_bar))/(2*G_bar*(1-nu_u_bar)*(1-nu_bar)))*F2_LT(z, s)


    # def transformed_term(s):
    #     G_bar =  (G*s)/(s+(G/mu_s_input))
    #     K_bar = K_d*s
    #     nu_bar = (3*K_bar-2*G_bar)/(2*(3*K_bar+G_bar))
    #     nu_u_bar = nu_bar

    #     return (1-2*nu_u_bar)/(2.*s*G_bar*(1-nu_u_bar))
    
    
    
    # term = inverse_laplace_transform(transformed_term, t_star) 
    # # nu_t = inverse_laplace_transform(transformed_nu, t_star) 

    # # print(G_t)

    # return P_0*L*term*(1-z_star)
    # # return ((P_0*L*(1-2*nu_u_t))/(2.*G_t*(1-nu_u_t)))*(1-z_star) + ((P_0*L*(nu_u_t-nu_t))/(2*G_t*(1-nu_u_t)*(1-nu_t)))*F2(z, t)

    
    return inverse_laplace_transform(transformed_u, t_star)

def write_displacement_PVE_file(filename, mu_s_input):

    # t_list = [1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    # t_list = np.asarray([1e6])*c/(4.*L**2)


    # t_list = np.asarray([1e6])*c/(4.*L**2)
    # z_list = np.arange(0, L+100, 100)/L
    z_list = np.arange(0, L+500, 500)/L


    # z_list = [0]
    # t_list = np.arange(0.01, 100, 0.02)

    f = open(filename, "w")
    f.write("t*,z*,uz\n")

    for t in t_list:
        for z in z_list:
            print("t*="+str(t)+", z*="+str(z))

            uz = displacement_PVE(z, t, mu_s_input)

            f.write(str(t)+","+str(z)+","+str(uz)+"\n")

    f.close()

def plot_displacement_PVE(filename, plot_pylith_PVE=False):

    df = pd.read_csv(filename, header=0, delimiter=',')
    # t_list = [1.4e-6, 1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    # color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    # t_list = [1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    # t_list = np.asarray([1e6])*c/(4.*L**2)
    color_list = ["tab:blue", "gold", "limegreen", "lightcoral"]
    color_list_pylith = ["darkblue", "orangered", "darkgreen", "darkred"]
    # print(0.01*((4. * L**2)/c))

    for t_star, cl, clp in zip(t_list, color_list, color_list_pylith):

        absolute_t = t_star*((4. * L**2)/c)

        maxwell_t = absolute_t*G/mu_s

        df_subset = df.loc[np.abs(df['t*'] - t_star)<1e-3]

        z = df_subset['z*'].values
        uz = df_subset['uz'].values

        plt.plot(uz / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"t/$\tau$="+str(round(maxwell_t, 3))+", PVE", lw=5, c=cl, alpha=0.8)
                                                                              
        plt.plot(displacement(z, t_star) / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(round(t_star, 3))+", PE", lw=7, c=cl, linestyle='dashed', alpha=0.5)

        if plot_pylith_PVE:

            absolute_t = t_star*((4. * L**2)/c)

            y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15.h5")
            plt.plot(uy_pylith / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith/L, label=r"t/$\tau$="+str(round(maxwell_t, 3))+", PyLith PVE", lw=5, c=clp, linestyle='solid', marker='o', alpha=0.8, markersize=8, mec='black', markevery=20)


            # y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain_phi=0_alpha=1E-5.h5")
            # y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/VE/terzaghi_quad_VE-domain_eta=1E15.h5")
            # plt.plot(uy_pylith_PE / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_PE/L, label=r"t/$\tau$="+str(round(maxwell_t, 3))+", PyLith VE", lw=7, c=clp, linestyle='dashed', alpha=0.5, marker='o', markersize=8, mec='black', markevery=20)


    plt.xlabel(r"$u_z  / \frac{-P_0L(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=28)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig("terzaghi_u_PVE_legend.png",  bbox_inches="tight")
    plt.show()

def check_PE_disp(t):

    z = np.arange(0, L+0.05, 0.05)/L
    t_star = t*c/(4.*L**2)
    print(t_star, t)

    df = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E15_phi=0_alpha=1E-5_select_times_new.csv", header=0, delimiter=',')
    # df = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E15_select_times.csv", header=0, delimiter=',')

    df_subset = df.loc[df['t*'] > 0.03]
    # df_subset = df.loc[df['t*'] == t]


    z_PVE = df_subset['z*'].values
    u_PVE = df_subset['uz'].values

    # print(u_PVE/((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))))

    plt.plot(displacement(z, t_star) / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(t_star)+", Analytic PE", lw=3)
    # plt.plot(displacement_PVE(z, t_star, 1E18) / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(t_star)+", Analytic PVE", lw=3, c="green")
    plt.plot(u_PVE / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z_PVE, label="t*="+str(t_star)+", Analytic PVE", lw=3, c="green")


    y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain_phi=0_alpha=1E-5.h5")
    # y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain.h5")
    plt.plot(uy_pylith_PE / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_PE/L, label="t*="+str(t_star)+", PyLith PE", lw=3, c='red', linestyle='dashed')

    y_pylith_VE, p_pylith, uy_pylith_VE, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/VE/terzaghi_quad_VE-domain_eta=1E15.h5")
    plt.plot(uy_pylith_VE / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_VE/L, label="t*="+str(t_star)+", PyLith VE", lw=3, c='green', linestyle='-.')


    y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15_phi=0_alpha=1E-5.h5")
    # y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15_long_times.h5")
    plt.plot(uy_pylith / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith/L, label="t*="+str(t_star)+", PyLith PVE", lw=3, c='black', linestyle='dotted')

    plt.xlabel(r"$u_z  / \frac{-P_0L(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=28)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    # plt.savefig("terzaghi_u_PVE_legend.png",  bbox_inches="tight")
    plt.show()

def plot_surface_disp():

    times = np.arange(1e5, 1e10, 1e5)

    non_dim_times = times*c/(4.*L**2)

    df_1E15 = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E15_long_times.csv", header=0, delimiter=',')
    df_1E13 = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E13_long_times.csv", header=0, delimiter=',')
    df_1E10 = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E10_long_times.csv", header=0, delimiter=',')


    t_analytic_PVE_1E15 = df_1E15['t*'].values
    u_analytic_PVE_1E15 = df_1E15['uz'].values

    t_analytic_PVE_1E13 = df_1E13['t*'].values
    u_analytic_PVE_1E13 = df_1E13['uz'].values

    t_analytic_PVE_1E10 = df_1E10['t*'].values
    u_analytic_PVE_1E10 = df_1E10['uz'].values


    t_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data_spatial(L, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain_long_times.h5")
    t_pylith_PVE, p_pylith, uy_pylith_PVE, trace_strain_pylith = read_pylith_data_spatial(L, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15_long_times.h5")
    t_pylith_PVE_1E17, p_pylith, uy_pylith_PVE_1E17, trace_strain_pylith = read_pylith_data_spatial(L, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E17_long_times.h5")
    t_pylith_PVE_1E20, p_pylith, uy_pylith_PVE_1E20, trace_strain_pylith = read_pylith_data_spatial(L, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E20_long_times.h5")


    plt.plot(non_dim_times, displacement(0, non_dim_times) / L, label="Analytic PE", lw=3)
    plt.plot(t_analytic_PVE_1E15, u_analytic_PVE_1E15 / L, label=r"Analytic PVE, $\mu_s=1E15$ Pa s", lw=3, c='green')
    plt.plot(t_analytic_PVE_1E13, u_analytic_PVE_1E13 / L, label=r"Analytic PVE, $\mu_s=1E13$ Pa s", lw=3, c='orange')
    plt.plot(t_analytic_PVE_1E10, u_analytic_PVE_1E10 / L, label=r"Analytic PVE, $\mu_s=1E10$ Pa s", lw=3, c='brown')


    plt.plot(t_pylith_PE*c/(4.*L**2), np.asarray(uy_pylith_PE) / L, label="PyLith PE", lw=3, c='red', linestyle='-')
    plt.plot(t_pylith_PVE*c/(4.*L**2), np.asarray(uy_pylith_PVE) / L, label=r"PyLith PVE, $\mu_s=1E15$ Pa s", lw=3, c='black', linestyle='--')
    plt.plot(t_pylith_PVE_1E17*c/(4.*L**2), np.asarray(uy_pylith_PVE_1E17) / L, label=r"PyLith PVE, $\mu_s=1E17$ Pa s", lw=3, c='black', linestyle='-.')
    plt.plot(t_pylith_PVE_1E20*c/(4.*L**2), np.asarray(uy_pylith_PVE_1E20) / L, label=r"PyLith PVE, $\mu_s=1E20$ Pa s", lw=3, c='black', linestyle=':')

    plt.xlabel("t*", fontsize=30)
    plt.ylabel(r"$u_z(0, t*)  / L$", fontsize=28)
    plt.xscale('log')
    # plt.gca().invert_yaxis()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.xlim([0.001, 100])
    plt.ylim([-3e-7, -0.7e-7])
    # plt.savefig("terzaghi_u_PVE_legend.png",  bbox_inches="tight")
    plt.show()

def strain_zz(z, t):

    # print((2./3.)*(-((P_0 * (1.0 - 2.0 * nu_u)) / (2.0 * G * (1.0 - nu_u)))))

    z_star = z
    t_star = t
    return -((P_0 * (1.0 - 2.0 * nu_u)) / (2.0 * G * (1.0 - nu_u))) \
        + ((P_0 * L * (nu_u - nu)) / (2.0 * G * (1.0 - nu_u) * (1.0 - nu))) * F3(z_star, t_star)

def dev_strain_zz_PVE(z, t):

    z_star = z
    t_star = t

    def transformed_strain(s):
        G_bar = G*s/(s+(G/mu_s))
        K_bar = K_d#K_d*s/(s+(G/mu_s))
        alpha_bar = alpha#alpha*s/(s+(G/mu_s))

        Ku_bar = K_bar + (K_fl*(K_sg-K_bar)**2)/(K_fl*(K_sg - K_bar) + phi*K_sg*(K_sg - K_fl))

        nu_bar = (3*K_bar-2*G_bar)/(2*(3*K_bar+G_bar))
        nu_u_bar = (3*Ku_bar-2*G_bar)/(2*(3*Ku_bar+G_bar))

        # def reduced_F3(ti):
        #     return F3_mpmath(z_star, ti)

        # F3_bar = laplace_transform(reduced_F3, s)

        return -((P_0*(1-2*nu_u_bar))/(2*s*G_bar*(1-nu_u_bar))) + ((P_0*L*(nu_u_bar-nu_bar))/(2*G_bar*(1-nu_u_bar)*(1-nu_bar)))*F3_LT(z_star, s)
        
    return inverse_laplace_transform(transformed_strain, t_star)

def write_dev_strain_zz_PVE_file(filename):

    # t_list = [1e-05, 0.001, 0.01, 0.05, 0.1, 0.2]
    # z_list = np.arange(0, L+0.5, 0.5)/L
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    z_list = np.arange(0, L+500, 500)/L

    f = open(filename, "w")
    f.write("t*,z*,e_zz\n")

    for t in t_list:
        for z in z_list:
            print("t*="+str(t)+", z*="+str(z))

            strain_zz = dev_strain_zz_PVE(z, t)

            f.write(str(t)+","+str(z)+","+str(strain_zz)+"\n")

    f.close()

def plot_strain_PVE(filename, plot_pylith_PVE=False):

    df = pd.read_csv(filename, header=0, delimiter=',')
    t_list = np.asarray([5e4, 5*5e4, 10*5e4, 1e6])*c/(4.*L**2)
    color_list = ["tab:blue", "gold", "limegreen", "lightcoral"]
    color_list_pylith = ["darkblue", "orangered", "darkgreen", "darkred"]

    for t_star, cl, clp in zip(t_list, color_list, color_list_pylith):

        df_subset = df.loc[np.abs(df['t*'] - t_star)<1e-3]

        z = df_subset['z*'].values
        dev_strain = df_subset['e_zz'].values
        strain = -dev_strain

        plt.plot(strain / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(round(t_star, 3))+", PVE", lw=5, c=cl, alpha=0.8)
                                                                              
        plt.plot(-strain_zz(z, t_star) / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(round(t_star, 3))+", PE", lw=7, c=cl, linestyle='dashed', alpha=0.5)

        if plot_pylith_PVE:

            absolute_t = t_star*((4. * L**2)/c)

            y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15.h5")
            plt.plot(trace_strain_pylith / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith/L, label="t*="+str(round(t_star, 3))+", PyLith PVE", lw=5, c=clp, linestyle='solid', marker='o', alpha=0.8, markersize=8, mec='black', markevery=20)


            # y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith_PE = read_pylith_data(absolute_t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain.h5")
            # plt.plot(trace_strain_pylith_PE / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_PE/L, label="t*="+str(round(t_star, 3))+", PyLith PE", lw=7, c=clp, linestyle='dashed', alpha=0.5, marker='o', markersize=8, mec='black', markevery=20)


    plt.xlabel(r"$\epsilon_{kk}  / \frac{-P_0(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=28)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig("terzaghi_strain_PVE_legend.png",  bbox_inches="tight")
    plt.show()


def check_PE_strain(t):

    z = np.arange(0, L+0.05, 0.05)/L
    t_star = t*c/(4.*L**2)
    print(t_star, t)

    df = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/dev_strain_PVE_eta_s=1E15_phi=0_alpha=1E-5_select_times.csv", header=0, delimiter=',')
    # df = pd.read_csv("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E15_select_times.csv", header=0, delimiter=',')

    df_subset = df.loc[df['t*'] > 0.03]

    z_PVE = df_subset['z*'].values
    e_PVE = df_subset['e_zz'].values

    plt.plot(-strain_zz(z, t_star) / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(t_star)+", Analytic PE", lw=3)
    plt.plot(-e_PVE / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z_PVE, label="t*="+str(t_star)+", Analytic PVE", lw=3, c="green")


    y_pylith_PE, p_pylith_PE, uy_pylith_PE, trace_strain_pylith_PE = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain_phi=0_alpha=1E-5.h5")
    # y_pylith_PE, p_pylith, uy_pylith_PE, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PE/terzaghi_quad-domain.h5")
    plt.plot(trace_strain_pylith_PE / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_PE/L, label="t*="+str(t_star)+", PyLith PE", lw=3, c='red', linestyle='dashed')

    y_pylith_VE, _, uy_pylith_VE, _ = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/VE/terzaghi_quad_VE-domain_eta=1E15.h5")
    strain_pylith_VE = np.gradient(uy_pylith_VE, y_pylith_VE)
    plt.plot(strain_pylith_VE / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith_VE/L, label="t*="+str(t_star)+", PyLith VE", lw=3, c='green', linestyle='-.')


    y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15_phi=0_alpha=1E-5.h5")
    # y_pylith, p_pylith, uy_pylith, trace_strain_pylith = read_pylith_data(t, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain_eta_s=1E15_test.h5")
    plt.plot(trace_strain_pylith / ((-P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), 1.0-y_pylith/L, label="t*="+str(t_star)+", PyLith PVE", lw=3, c='black', linestyle='dotted')

    plt.xlabel(r"$\epsilon_{zz}  / \frac{-P_0(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=28)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    # plt.savefig("terzaghi_u_PVE_legend.png",  bbox_inches="tight")
    plt.show()

def plot_stress_PVE(strain_filename, p_filename):

    df_strain = pd.read_csv(strain_filename, header=0, delimiter=',')
    df_pressure = pd.read_csv(p_filename, header=0, delimiter=',')
    t_list = [1e-5, 0.001, 0.01, 0.05, 0.1, 0.2]
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for t, c in zip(t_list, color_list):

        df_strain_subset = df_strain.loc[df_strain['t*'] == t]
        df_pressure_subset = df_pressure.loc[df_pressure['t*'] == t]

        z = df_strain_subset['z*'].values
        dev_strain = df_strain_subset['e_zz'].values
        p = df_pressure_subset['p'].values
        strain = dev_strain#+strain_zz(z, t)/3

        stress = ((2*G*nu)/(1-2*nu))*strain + 2*G*strain - alpha*p

        plt.plot(stress / P_0, z, label="t*="+str(t)+", PVE", lw=3, c=c)
                                                                              
        plt.plot(stress_zz(z, t) / P_0, z, label="t*="+str(t)+", PE", lw=3, linestyle='dashed')
        


    plt.xlabel(r"$\sigma_{zz}  / P_0$", fontsize=28)
    plt.ylabel("z*", fontsize=30)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig("terzaghi_stress_PVE_legend.png",  bbox_inches="tight")
    plt.show()


def stress_zz(z, t):

    return (2.*G*nu)/(1-2.*nu)*strain_zz(z, t) + 2*G*strain_zz(z, t) - alpha*pressure(z, t)



def stress_xx(z, t):

    return (2.*G*nu)/(1-2.*nu)*strain_zz(z, t) - alpha*pressure(z, t)

def F1(z_star, t_star):
    F1 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F1 += 4. / (m * np.pi) * np.sin(0.5 * m * np.pi * z_star) * np.exp(-(m * np.pi)**2 * t_star)
    return F1

# version of F1 with mpmath data types that can be Laplace transformed/inverse transformed
def F1_mpmath(z_star, t_star):
    F1 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F1 += 4. / (m * np.pi) * mpmath.sin(0.5 * m * np.pi * z_star) * mpmath.exp(-(m * np.pi)**2 * t_star)
    return F1

# Laplace transformed F1
def F1_LT(z_star, s):
    F1=0
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F1 += 4. / (m * np.pi) * mpmath.sin(0.5 * m * np.pi * z_star) * (1./(s+m**2*np.pi**2))
    return F1


def F1_dot(z_star, t_star):
    F1_dot = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F1_dot += (-4.*m*np.pi)  * np.sin(0.5 * m * np.pi * z_star) * np.exp(-(m * np.pi)**2 * t_star)
    return F1_dot

def F2(z_star, t_star):

    F2 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F2 += (8. / (m * np.pi)**2) * np.cos(0.5 * m * np.pi * z_star) * (1. - np.exp(-(m * np.pi)**2 * t_star))
    return F2

def F2_LT(z_star, s):

    F2 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F2 += (8. / (m * np.pi)**2) * mpmath.cos(0.5 * m * np.pi * z_star) * (1./s - (1./(s+m**2*np.pi**2)))
    return F2

def F2_mpmath(z_star, t_star):

    F2 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F2 += (8. / (m * np.pi)**2) * mpmath.cos(0.5 * m * np.pi * z_star) * (1. - mpmath.exp(-(m * np.pi)**2 * t_star))
    return F2

def F3(z_star, t_star):

    F3 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F3 += (-4.0 / (m * np.pi * L)) * np.sin(0.5 * m * np.pi * z_star) * (1.0 - np.exp(-(m * np.pi)**2 * t_star))
    return F3

def F3_mpmath(z_star, t_star):

    F3 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F3 += (-4.0 / (m * np.pi * L)) * mpmath.sin(0.5 * m * np.pi * z_star) * (1.0 - mpmath.exp(-(m * np.pi)**2 * t_star))
    return F3

def F3_LT(z_star, s):

    F3 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F3 += (-4.0 / (m * np.pi * L)) * mpmath.sin(0.5 * m * np.pi * z_star) * (1.0/s - (1./(s+m**2*np.pi**2)))
    return F3


def F3_dot(z_star, t_star):

    F3_dot = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F3_dot += (-4.0 / (m * np.pi * L)) * np.sin(0.5 * m * np.pi * z_star) * (1.0 + m**2*np.pi**2*np.exp(-(m * np.pi)**2 * t_star))
    return F3_dot


def F4(z_star, t_star):

    F4 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F4 += (-4.0 / (m * np.pi * L)) * np.sin(0.5 * m * np.pi * z_star) * (t_star + ((np.exp(-(m * np.pi)**2 * t_star) + 1)/(m**2*np.pi**2)))
    return F4

def F5(z_star, t_star):
    F5 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F5 += -4. / (m * np.pi) * np.sin(0.5 * m * np.pi * z_star) * ((np.exp(-(m * np.pi)**2 * t_star) + 1)/m**2*np.pi**2)
    return F5

def F6(z_star, t_star):

    F6 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F6 += (8.0 / (m**2 * np.pi**2 )) * (np.cos(0.5 * m * np.pi * z_star)-1) * (t_star + ((np.exp(-(m * np.pi)**2 * t_star) + 1)/(m**2*np.pi**2)))
    return F6

def F7(z_star, t_star):

    F7 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F7 += (8.0 / (m**2 * np.pi**2 )) * (np.cos(0.5 * m * np.pi * z_star)-1) * (((np.exp(-(m * np.pi)**2 * t_star) + 1)/(m**2*np.pi**2)))
    return F7

def F8(z_star, t_star):

    F8 = 0.
    for m in np.arange(1, 2 * ITERATIONS + 1, 2):
        F8 -= (8.0 / (m**2 * np.pi**2 )) * (np.cos(0.5 * m * np.pi * z_star)-1) * np.exp(-m**2*np.pi**2*t_star)
    return F8

def laplace_transform(f, s):

    def f_full(t):
        # return f(t)*np.exp(-float(s)*t)
        return f(t)*mpmath.exp(-s*t)

    
    # return integrate.quad(f_full, 0, np.inf)[0]
    return mpmath.quad(f_full, [0, np.inf])


def inverse_laplace_transform(f, t):
    mpmath.dps = 30; mpmath.pretty = True

    # return mpmath.invertlaplace(f, t, method='talbot')
    return mpmath.invertlaplace(f, t)


def test_laplace_transforms():

    # alpha=250
    z=1
    def initial_time_space(t):
        # return mpmath.exp(-alpha*t)
        return F1_mpmath(z, t)
    
    def initial_time_space_plot(t):
        # return np.exp(-alpha*t)
        return -(P_0*eta/(G*S))*F1(z, t)
        # return F3(z, t)
    
    def analytic_transform(s):
        # print(s)
        mu_s_prime=1e15
        G_bar = (s*G*mu_s_prime)/(s*mu_s_prime + G) #G*s/(s+(G/mu_s_prime))
        K_bar = (s*K_u*mu_s_prime)/(s*mu_s_prime + K_u) #K_d*s/(s+(G/mu_s_prime))
        alpha_bar = (s*alpha*mu_s_prime)/(s*mu_s_prime + alpha) #alpha*s/(s+(G/mu_s_prime))
        K_sg_bar = K_bar/(1.0-alpha_bar)

        # K_sg_bar = K_sg*s/(s+(G/mu_s_prime))
        # K_fl_bar = K_fl
        # K_bar = 1./((1./((1.-phi)*K_sg_bar)) + (1./((1.-phi)**3*K_fl_bar)))
        # alpha_bar = 1. - ((1.-phi)**3*K_fl_bar)/(K_sg_bar + (1.-phi)**2*K_fl_bar)


        eta_bar = (3.0 * alpha_bar * G_bar) / (3.0 * K_bar + 4.0 * G_bar)
        inverse_M_bar = (alpha_bar-phi)/K_sg_bar + phi/(K_fl)
        S_bar = inverse_M_bar + 3*alpha_bar**2/(3*K_bar+4*G_bar)
        return -(P_0*eta_bar/(G_bar*S_bar))*F1_LT(z, s)
        # return F3_LT(z, s)

        # return 1/(s+alpha)

    
    # s = np.arange(0.01, 10, 1)
    # # s=np.asarray([1e-5, 0.001])
    # transform = []
    # for si in s:
    #     transform.append(laplace_transform(initial_time_space, si))

    # # plt.plot(s, analytic_transform, c='black', lw=5)

    
    # plt.plot(s, analytic_transform(s), linestyle='solid', c="black", lw=5, label="analytic")
    # plt.plot(s, transform, linestyle='dashed', lw=5, c="blue", label="numeric")
    # plt.xlabel("s", fontsize=20)
    # plt.ylabel(r"$F(s)=F1^*(z, s)$", fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.yscale('log')
    # plt.grid()
    # plt.legend(fontsize=20)
    # plt.show()
    
    # t = np.arange(0.01, 20, 1)
    t = np.asarray([1e-5, 1e-3, 0.05, 0.1, 0.2, 0.5])
    # t=np.asarray([1e-5, 0.001])
    inverse_transform = []
    for ti in t:
        # print(ti)
        inverse_transform.append(inverse_laplace_transform(analytic_transform, ti))

    
    plt.plot(t, initial_time_space_plot(t) * ((G*S)/(-P_0*eta)),  linestyle='solid', c="black", lw=5, label="analytic/PE")
    plt.plot(t, np.asarray(inverse_transform) * ((G*S)/(-P_0*eta)), linestyle='dashed', lw=5, c="blue", label="numeric/PVE")
    # plt.plot(t, initial_time_space_plot(t),  linestyle='solid', c="black", lw=5, label="analytic/PE")
    # plt.plot(t, np.asarray(inverse_transform), linestyle='dashed', lw=5, c="blue", label="numeric/PVE")
    plt.xlabel(r"$t$", fontsize=20)
    # plt.ylabel(r"$f(t)=F3(z, t)$", fontsize=20)
    plt.ylabel(r"$f(t)=p(z, t)$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.yscale('log')
    plt.grid()
    plt.legend(fontsize=20)
    plt.show()


def read_pylith_data(t, path):

    # path = "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain.h5"

    with h5py.File(path, "r") as f:
        group_geometry = f['geometry']
        points = group_geometry['vertices'] #shape: point_num, xyz
        x = points[:][:,0]
        y = points[:][:,1]

        group_vert_fields = f['vertex_fields']
        times = f['time'][:,0,0]

        # get time index closest to t
        ti = np.argmin(np.abs(times - t))
        # print(ti, times[ti])
        
        pressures = group_vert_fields['pressure'][ti][:][:,0]
        uy = group_vert_fields['displacement'][ti][:][:,1]
        trace_strain = group_vert_fields['trace_strain'][ti][:][:,0]

        #interpolate data 
        #get points to interpolate
        y_interp = np.linspace(min(y), max(y), 2*len(y))
        x_interp = np.zeros(len(y_interp))

        p_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(pressures), np.asarray([x_interp, y_interp]).T, method='cubic')
        uy_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(uy), np.asarray([x_interp, y_interp]).T, method='cubic')
        trace_strain_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(trace_strain), np.asarray([x_interp, y_interp]).T, method='cubic')

        return y_interp, p_interp, uy_interp, trace_strain_interp

def read_pylith_data_spatial(z, path):

    # path = "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/terzaghi_quad-domain.h5"

    with h5py.File(path, "r") as f:
        group_geometry = f['geometry']
        points = group_geometry['vertices'] #shape: point_num, xyz
        x = points[:][:,0]
        y = points[:][:,1]

        group_vert_fields = f['vertex_fields']
        times = f['time'][:,0,0]

        # get time index closest to t
        # yi = np.argmin(np.abs(y - z))
        yi = np.where((x == 0) & (y == z))[0][0]

        
        pressures = group_vert_fields['pressure'][:][:][:,yi,0]
        uy = group_vert_fields['displacement'][:][:][:,yi,1]
        trace_strain = group_vert_fields['trace_strain'][:][:][:,yi,0]

        return times, pressures, uy, trace_strain
    

def test_analytic_LT():

    s, t = sympy.symbols('s, t')
    G, K, a, mu_s_sym, phi_sym, K_f, K_s, P_sym, L_sym, z = sympy.symbols('G, K, a, mu_s, phi, K_f, K_s, P, L, z', real=True)

    G_bar = (G*s)/(s + G/mu_s_sym)
    K_bar = (K*s)/(s + G/mu_s_sym)
    a_bar = (a*s)/(s + G/mu_s_sym)
    K_s_bar = K_bar/(1.0 - a_bar)
    M_bar = 1/((phi_sym/K_f) + ((a_bar-phi_sym)/K_s))
    K_u_bar = K_bar + a_bar**2 * M_bar
    nu_bar = (3*K_bar - 2*G_bar)/(2*(3*K_bar + G_bar))
    nu_u_bar = (3*K_u_bar - 2*G_bar)/(2*(3*K_u_bar + G_bar))

    expression = (P_sym*L_sym*(1-2*nu_u_bar)/(2*s*G_bar*(1-nu_u_bar))) * (1-z)

    answer = sympy.inverse_laplace_transform(expression, s, t)
    print(answer)





# make plots
z = np.arange(0, L+0.05, 0.05)/L
# dt = ((4. * L**2)/c)

# read_pylith_data_spatial(L, "/home/grantblock/Research/PylithPVE/local_fullscale_data/terzaghi/results/PVE/paraview/terzaghi_quad-domain.h5")

# test_laplace_transforms()
# write_pressure_PVE_file("pressure_PVE_eta_s=1E15_select_times.csv")
# write_pressure_PVE_file("pressure_PVE_eta_s=1E15_compare_pylith.csv")
# write_displacement_PVE_file("disp_PVE_eta_s=1E15_phi=0_alpha=1E-5_select_times_new.csv", 1E15)
# write_displacement_PVE_file("disp_PVE_eta_s=1E15_compare_pylith.csv", 1E15)
# write_dev_strain_zz_PVE_file("dev_strain_PVE_eta_s=1E15_compare_pylith.csv")

# for i in range(2):
#     for j in range(2):
#         print(((i * 2 + i) * 2 + j) * 2 + j)
#         print(((i * 2 + j) * 2 + j) * 2 + i)
#         print("----")

# for i in range(2):
#     print(i*2+i)

# check_PE_pressure(1e6)
# check_PE_disp(1e6)
# check_PE_strain(1e6)

# plot_surface_disp()
# test_analytic_LT()

# plot_pressure_PVE("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/pressure_PVE_eta_s=1E15_compare_pylith.csv", plot_pylith_PVE=True)
# plot_displacement_PVE("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/disp_PVE_eta_s=1E15_compare_pylith.csv", plot_pylith_PVE=True)
plot_strain_PVE("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/dev_strain_PVE_eta_s=1E15_compare_pylith.csv", plot_pylith_PVE=True)
# plot_stress_PVE("~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/dev_strain_PVE_eta_s=1E10.csv", "~/Research/PylithPVE/local_fullscale_data/terzaghi/scripts/pressure_PVE_eta_s=1E10.csv")

# times = [1*dt, 10*dt, 100*dt, 1000*dt]
# colors = ['blue', 'orange', 'green', 'brown', 'red', 'purple']
# for t, cl in zip(times, colors):
#     plt.plot((2./3.)*strain_zz(z, t), z, label="PE, t*="+str(t/dt), lw=3, color=cl)

# # for t, cl in zip(times, colors):
# #     plt.plot((dev_strain_zz_PVE(z, t)+(1./3.)*strain_zz(z,t))/(-(P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="PVE, non-iterative, t*="+str(t), lw=3, linestyle='dashed', color=cl)

# for t, cl in zip(times, colors):
#     plt.plot(iterative_PVE_dev_strain_zz_int(z, t, dt, eta_s=1e5), z, label="PVE, iterative, t*="+str(t/dt), lw=3, linestyle='dotted', color=cl)

# # plt.xlabel(r"$\epsilon'_{zz}/\frac{-P_0(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=30)
# plt.xlabel(r"$\epsilon'_{zz}$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# # plt.xscale('log')
# plt.legend(fontsize=15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()
# times = [1e-5, 0.001, 0.01, 0.05, 0.1, 0.2]

# for t in times:
#     plt.plot(pressure(z, t) * ((G*S)/(-P_0*eta)), z, label="t*="+str(t), lw=3)

# plt.xlabel(r"$p / \frac{P_0\eta}{GS}$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# for t in times:
#     plt.plot(displacement(z, t) / ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label="t*="+str(t), lw=3)

# plt.xlabel(r"$U_z / \frac{-P_0L(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# # plt.savefig("terzaghi_disp_legend.png",  bbox_inches="tight")
# plt.show()

# z = np.arange(0.001, L+0.05, 0.05)/L

# times=[0.0]

# for t in times:
#     plt.plot(stress_zz(z, t)/P_0, z, label="t*="+str(t), lw=3)

# plt.xlabel(r"$\sigma_{zz}/P_0$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# print(2*G*dev_strain_zz_PVE(0, 0))

# for t in times:
#     plt.plot(stress_xx(z, t)/P_0, z, label="t*="+str(t), lw=3)

# plt.xlabel(r"$\sigma_{xx}/P_0$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# times = [0.001 ,0.01, 0.05, 0.1, 0.2]
# for t in times:
#     plt.plot(F1(z, t), z, label="t*="+str(t), lw=3)


# plt.xlabel(r"$F_1$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # plt.xscale('log')
# plt.grid()
# plt.show()

# times = [0.001 ,0.01, 0.05, 0.1, 0.2]
# for t in times:
#     plt.plot(F2(z, t), z, label="t*="+str(t), lw=3)


# plt.xlabel(r"$F_2$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # plt.xscale('log')
# plt.grid()
# plt.show()


colors=['blue', 'orange', 'green', 'red', 'purple', 'brown']
# itr = 0
# for t in times:
#     plt.plot(dev_strain_zz_PVE(z, t, eta_s=1e10)/(-(P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=3.3E9$ s,t*="+str(t), lw=3, c=colors[itr])
#     itr += 1
# itr=0
# for t in times:
#     plt.plot(dev_strain_zz_PVE(z, t, eta_s=1e3)/(-(P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=3.3E2$ s,t*="+str(t), lw=3, linestyle='dashed', c=colors[itr])
#     itr+=1

# plt.xlabel(r"$\epsilon'^{PVE}_{zz}/\frac{-P_0(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=25)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# # plt.legend(fontsize=15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# itr = 0
# for t in times:
#     plt.plot(dev_strain_xx_PVE(z, t, eta_s=1e10)/(-(P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=3.3E9$ s,t*="+str(t), lw=3, c=colors[itr])
#     itr+=1

# itr = 0
# for t in times:
#     plt.plot(dev_strain_xx_PVE(z, t, eta_s=1e3)/(-(P_0*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=3.3E2$ s,t*="+str(t), lw=3, linestyle='dashed', c=colors[itr])
#     itr+=1

# plt.xlabel(r"$\epsilon'^{PVE}_{xx}/\frac{-P_0(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=25)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# # plt.legend(fontsize=15)
# # plt.legend(fontsize=25, bbox_to_anchor=(1.15, 1.0), loc='upper left')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# # plt.savefig("terzaghi_PVE_dev_strain_legend.png",  bbox_inches="tight")
# plt.show()

# for t in times:
#     plt.plot(int_pressure(z, t), z, label="t*="+str(t), lw=3)

# plt.xlabel(r"$\int_0^tpdt'$", fontsize=30)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # plt.xscale('log')
# plt.grid()
# plt.show()

# colors=['blue', 'orange', 'green', 'red', 'purple', 'brown']
# itr = 0
# for t in times:
#     plt.plot(PVE_disp(z, t, eta_s=1e10)/ ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=3.3E9$ s,t*="+str(t), lw=3, c=colors[itr])
#     itr += 1
# itr=0
# for t in times:
#     plt.plot(PVE_disp(z, t, eta_s=1e2)/ ((-P_0*L*(1.-2.*nu_u))/(2.*G*(1.-nu_u))), z, label=r"$\tau=33$ s,t*="+str(t), lw=3, linestyle='dashed', c=colors[itr])
#     itr+=1

# plt.xlabel(r"$u_z^{PVE}/\frac{-P_0L(1-2\nu_u)}{2G(1-\nu_u)}$", fontsize=25)
# plt.ylabel("z*", fontsize=30)
# plt.gca().invert_yaxis()
# plt.legend(fontsize=15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# times = np.arange(1e-6, 10, 0.05)
# plt.plot(times, PVE_disp_test(0, np.asarray(times), eta_s=10))
# plt.xscale('log')
# plt.show()
