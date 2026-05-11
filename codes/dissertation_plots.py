import AnalyzeModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from scipy import interpolate
from scipy.optimize import curve_fit


#--------------------Plot functions--------------------------------------#

# Plot velocity profiles normalized by dp/dt scaling
def dpdt_scaling_plot(model_list, time):

    d_uc = 10e3

    # plot scaled velocities
    linestyles=['solid', 'dashed', 'dotted']
    for model, ls in zip(model_list, linestyles):
        
        scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
        
        # get velocity profiles at given time
        model_time = model.get_timesteps(time)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)

        # plot scaled velocities
        plt.plot(x/1e3, vel_z*model.ms_to_mmyr*scaling_param, lw=10, ls=ls, label="|dP/dt|="+str(model.model.Delta_P/1e3)+" kPa/yr")

    # set up plot parameters
    plt.xlabel("Distance From Center (km)", fontsize=30)
    plt.ylabel(r"$c_{vp}V_z$", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([-50, 50])
    # plt.legend(fontsize=30)
    plt.grid()

    plt.show()

    # plot unscaled velocities
    center_vel_list = []
    dpdt_list = []
    for model in model_list:
        # get velocity profiles at given time
        model_time = model.get_timesteps(time)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
        center_vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr)
        dpdt_list.append(model.model.Delta_P/1e3)

        # plot velocities
        plt.plot(x/1e3, vel_z*model.ms_to_mmyr, lw=10, label="|dP/dt|="+str(model.model.Delta_P/1e3)+" kPa/yr")

    # set up plot parameters again
    plt.xlabel("Distance From Center (km)", fontsize=30)
    plt.ylabel(r"$V_z$ (mm/yr)", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([-50, 50])
    plt.legend(fontsize=30)
    plt.grid()

    plt.show()

    # Plot max velocity vs. dpdt
    plt.xlabel("|dP/dt| (kPa/yr)", fontsize=30)
    plt.ylabel(r"$V_z(r=0)$ (mm/yr)", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.plot(dpdt_list, center_vel_list, lw=5, c='black')
    plt.scatter(dpdt_list, center_vel_list, c='black', s=100, zorder=10)
    plt.show()

def visc_scaling_plot(model_list, visc_list, shear_mod_list, depth_list, time):

    rho_CR = 2500 #kg/m^3

    # plot scaled velocities
    linestyles=['solid', 'dashed', 'dotted']
    for model, eta, mu, depth, ls in zip(model_list, visc_list, shear_mod_list, depth_list, linestyles):

        # scaling_param = (eta/(mu*depth))*1e3*3.17098e-8 # in yr/mm
        kinematic_visc = eta/rho_CR
        scaling_param = np.sqrt(eta)

        # get velocity profiles at given time
        model_time = model.get_timesteps(time)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)

        # plot scaled velocities
        plt.plot(x/1e3, vel_z*model.ms_to_mmyr*scaling_param, lw=10, ls=ls, label=r"$\eta_{CR}$="+str(round(eta,3))+" Pa s")

    # set up plot parameters
    plt.xlabel("Distance From Center (km)", fontsize=30)
    plt.ylabel(r"$c_2V_z$", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([-50, 50])
    plt.legend(fontsize=30)
    plt.grid()

    plt.show()

    # plot unscaled velocities
    center_vel_list = []
    eta_list = []
    for model, eta in zip(model_list, visc_list):
        # get velocity profiles at given time
        model_time = model.get_timesteps(time)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
        center_vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr)
        eta_list.append(eta)

        # plot velocities
        plt.plot(x/1e3, vel_z*model.ms_to_mmyr, lw=10, label=r"$\eta_{CR}$="+str(round(eta,3))+" Pa s")

    # set up plot parameters again
    plt.xlabel("Distance From Center (km)", fontsize=30)
    plt.ylabel(r"$V_z$ (mm/yr)", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([-50, 50])
    plt.legend(fontsize=30)
    plt.grid()

    plt.show()

    # Plot max velocity vs. eta
    plt.xlabel(r"$\eta_{CR}$ (Pa s)", fontsize=30)
    plt.ylabel(r"$V_z(r=0)$ (mm/yr)", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.plot(eta_list, center_vel_list, lw=5, c='black')
    plt.scatter(eta_list, center_vel_list, c='black', s=100, zorder=10)
    plt.show()

# Plot pressure function from file
def plot_pressure_func(model, pressure_func_file):

    # read in pressure function
    df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
    pressure_times = df['times']
    pressure_values = df['pressures']*model.model.P0/1e3

    # set up plotting parameters
    plt.xlabel("Time (yrs)", fontsize=30)
    plt.ylabel("Pressure (kPa)", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([model.model.spinup_time-10, model.model.spinup_time+model.model.T*model.model.cycles+10])
    plt.grid()

    # plot
    plt.plot(pressure_times, pressure_values, c="black", lw=10)
    plt.show()

# Plot profiles of a model at different times
def plot_profile_time(model, times, labels, pt_norm=True, plot_source_lines=False, CR_lines=None, inner_CR_lines=None, x_lim=[-30, 30], theta=None):

    # get base profile
    (x0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(0, theta=None)

    # loop through given times
    for t, l in zip(times, labels):

        # get model output
        model_time = model.get_timesteps(t)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        if pt_norm:
            d_uc=10e3
            scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm

             # plot velocity profile
            plt.plot(x0/1e3, vel_z*model.ms_to_mmyr*scaling_param, lw=10, label=l)

        else:
            plt.plot(x0/1e3, vel_z*model.ms_to_mmyr, lw=10, label=l)
            scaling_param = 1

    # plot vertical lines to represent source and/or CR edge
    if plot_source_lines:
        plt.axvline(x=model.r_x, c='black', linestyle='dotted', lw=7)
        plt.axvline(x=-model.r_x, c='black', linestyle='dotted', lw=7)

    if CR_lines != None:
        plt.axvline(x=CR_lines[0], c='black', linestyle='dashed', lw=7)
        plt.axvline(x=CR_lines[1], c='black', linestyle='dashed', lw=7)


    # plot settings
    plt.xlabel("Distance From Center (km)", fontsize=30)
    if pt_norm:
        plt.ylabel(r"$c_{vp}V_z$", fontsize=30)
    else:
        plt.ylabel(r"$V_z$ (mm/yr)", fontsize=30)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(x_lim)
    # plt.legend(fontsize=40)
    plt.grid()

    plt.show()

# slimmed down function to plot the ground surface velocity
def plot_groundsurf_vel(model, time, pt_norm=True, vel_lim=None, source=True, CR=None, CR_inner=None, x_lim=None, y_lim=None):

     #first need to get model time
    model_time = model.get_timesteps(time)

    #get the model file path
    path = "../../../Yellowstone/"+str(model.model.path)+"/"+model.model.path[4:]+"-groundsurf.h5"
       
    #prepare lists for getting data from hdf5 files
    with h5py.File(path, "r") as f:
        group_geometry = f['geometry']
        points = group_geometry['vertices'] #shape: point_num, xyz
        x = points[:][:,0]
        y = points[:][:,1]

        group_vert_fields = f['vertex_fields']            
        velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        vel_z = velocities[model_time][:][:,2]

    #interpolate
    grid_x, grid_y = np.mgrid[-model.mesh_width:model.mesh_width:1000j, -model.mesh_width:model.mesh_width:1000j]
    z = interpolate.griddata((x/1e3,y/1e3), vel_z, (grid_x/1e3, -grid_y/1e3), method='cubic')

    # make plot
    plt.figure()
    color_map='seismic'

    if pt_norm:
        d_uc=10e3
        scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
    else:
        scaling_param = 1

    from matplotlib import colors
    if vel_lim == None:
        divnorm=colors.TwoSlopeNorm(vmin=-20.*scaling_param, vcenter=0., vmax=60*scaling_param)
    else: 
        divnorm=colors.TwoSlopeNorm(vmin=vel_lim[0]*scaling_param, vcenter=0., vmax=vel_lim[1]*scaling_param)


    im = plt.imshow(z.T*model.ms_to_mmyr*scaling_param, extent=(-model.mesh_width/1e3,model.mesh_width/1e3,-model.mesh_width/1e3,model.mesh_width/1e3),
                    cmap=color_map, norm=divnorm)

    # cbar = plt.colorbar(im, orientation='horizontal')
    cbar = plt.colorbar(im)
    cbar.set_label(label=r"$c_{vp}V_z$", size=40)
    cbar.ax.tick_params(labelsize=30)   
    plt.xlabel("X [km]", fontsize=40)
    plt.ylabel("Y [km]", fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=40)

     # plot source
    if source:
       source_x_points, source_y_points = AnalyzeModel.get_CR_points(model.r_x, model.r_y)
       plt.plot(source_x_points+model.x_off/1e3, source_y_points+model.y_off/1e3, c='black', linestyle='dashdot', linewidth=4.5, zorder=3)

    # plot CR
    if CR != None:
        CR_x_points, CR_y_points = AnalyzeModel.get_CR_points(CR[0], CR[1])
        plt.plot(CR_x_points, CR_y_points, c='black', linestyle='dashed', linewidth=5)
            
    # plot inner CR
    if CR_inner != None:
        CR_x_inner, CR_y_inner = AnalyzeModel.get_CR_points(CR_inner[0], CR_inner[1])
        plt.plot(CR_x_inner, CR_y_inner, c='black', linestyle='dashdot', linewidth=3)

    # set plot limits
    if x_lim != None:
        plt.xlim(x_lim)
    if y_lim != None:
        plt.ylim(y_lim)

    plt.show()

# Function to take lists of related models differing in source dimension and depth and plot their trend of depth/width vs. velocity
def plot_depth_width_trend(model_lists, depth_lists, times_list, labels, pt_norm=True, depth_norm=False, CR_norm=None,
                           elastic_models=None, elastic_depths=None, elastic_times=None, elastic_label=None,
                           visc_models=None, visc_depths=None, visc_times=None, visc_label=None):

    # iterate through each model list
    for models, depths, l in zip(model_lists, depth_lists, labels):

        # for each model in the list, get the max magnitude vel
        # in the time span
        max_vels = []
        for model, times in zip(models, times_list):

            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to get the max vel
            # print(model.model.model_name, times)
            vel_list = []
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr*scaling_param)

            max_vels.append(-min(np.asarray(vel_list)))
        
        # plot max vel with depth
        if depth_norm:
            plt.plot(np.asarray(depths)/models[0].r_x, max_vels, lw=7)
            plt.scatter(np.asarray(depths)/models[0].r_x, max_vels, s=200, label=l, edgecolors='black', zorder=10)
        else:
            plt.plot(depths, max_vels, lw=7)
            plt.scatter(depths, max_vels, s=200, label=l, edgecolors='black', zorder=10)

    # if elastic models are given plot them too
    if elastic_models != None:

        max_vels = []
        for model, times in zip(elastic_models, elastic_times):
            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to get the max vel
            # print(model.model.model_name, times)
            vel_list = []
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr*scaling_param)

            max_vels.append(-min(np.asarray(vel_list)))

        plt.plot(elastic_depths, max_vels, lw=7, c='gray', linestyle='dotted')
        plt.scatter(elastic_depths, max_vels, s=200, c='gray', label=elastic_label, edgecolors='black', zorder=10)

        
    # if visc models are given plot them too    
    if visc_models != None:

        max_vels = []
        for model, times in zip(visc_models, visc_times):
            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to get the max vel
            # print(model.model.model_name, times)
            vel_list = []
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr*scaling_param)

            max_vels.append(-min(np.asarray(vel_list)))
            # max_vels.append(max(np.asarray(vel_list)))
                

        plt.plot(visc_depths, max_vels, lw=7, c='rosybrown', linestyle='dotted')
        plt.scatter(visc_depths, max_vels, s=200, c='rosybrown', label=visc_label, edgecolors='black', zorder=10)
    
    # plot parameters
    plt.xlabel(r"$d_s$ (km)", fontsize=30)
    if pt_norm:
        plt.ylabel(r"$c_{vp}|V_z|$", fontsize=30)
    else:
        plt.ylabel(r"$|V_z|$ (mm/yr)", fontsize=30)

    plt.xticks(fontsize=25)
    plt.yscale('log')
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left',fontsize=30)
    plt.grid()

    file_name = "/home/grantblock/Research/SMBPylith/Figures/depth_vel_trend"
    plt.savefig(file_name, bbox_inches="tight")

    plt.show()

    # Do the same iteration as before, but this time calculate the max FWHM value
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if CR_norm != None:
        minor_norm = CR_norm[0]
        major_norm = CR_norm[1]
    else:
        minor_norm = 1
        major_norm = 1
    for models, depths, l, color in zip(model_lists, depth_lists, labels, colors):

        # for each model in the list, get the FWHM at the max magnitude vel
        # in the time span
        max_widths = []
        max_widths_major = []
        for model, times in zip(models, times_list):

            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
            min_vel = np.infty
            FWHM = 0
            FWHM_major = 0
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                
                if min(vel_z) < min_vel:
                    min_vel = min(vel_z)

                    idx = np.argmin(np.abs(vel_z[x<0] - min_vel/2))
                    idx_major = np.argmin(np.abs(vel_z_major[x<0] - min_vel/2))

                    FWHM = abs(2*x[x<0][idx])
                    FWHM_major = abs(2*x[x<0][idx_major])

            max_widths.append((FWHM/1e3)/minor_norm)
            max_widths_major.append((FWHM_major/1e3)/major_norm)
        # print("minor", max_widths)
        # print("major", max_widths_major)
        # print("depths", depths)

        if depth_norm:
            depths = np.asarray(depths)/models[0].r_x
        
        # plot FWHM at max vel with depth
        if color == "tab:blue":
            plt.plot(depths, max_widths, lw=7, label="Minor Axis Profile", c=color)
            plt.scatter(depths, max_widths, s=200, edgecolors='black', zorder=10, c=color)

            plt.plot(depths, max_widths_major, lw=7, linestyle='dashed', label="Major Axis Profile", c=color)
            plt.scatter(depths, max_widths_major, s=200, edgecolors='black', zorder=10, c=color)
        else:
            plt.plot(depths, max_widths, lw=7, c=color)
            plt.scatter(depths, max_widths, s=200, edgecolors='black', zorder=10, c=color)

            plt.plot(depths, max_widths_major, lw=7, linestyle='dashed', c=color)
            plt.scatter(depths, max_widths_major, s=200, edgecolors='black', zorder=10, c=color)

    # Check if there are elastic models to plot
    if elastic_models != None:
        max_widths = []
        max_widths_major = []

        for model, times in zip(elastic_models, elastic_times):
            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
            min_vel = np.infty
            FWHM = 0
            FWHM_major = 0
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                
                if min(vel_z) < min_vel:
                    min_vel = min(vel_z)

                    idx = np.argmin(np.abs(vel_z[x<0] - min_vel/2))
                    idx_major = np.argmin(np.abs(vel_z_major[x<0] - min_vel/2))

                    FWHM = abs(2*x[x<0][idx])
                    FWHM_major = abs(2*x[x<0][idx_major])

            max_widths.append((FWHM/1e3)/minor_norm)
            max_widths_major.append((FWHM_major/1e3)/major_norm)

        plt.plot(elastic_depths, max_widths, lw=7, c='gray', linestyle='dotted', label="Elastic HS Minor Axis Profile", alpha=0.7)
        plt.scatter(elastic_depths, max_widths, s=200, edgecolors='black', zorder=10, c='gray', alpha=0.5)

        plt.plot(elastic_depths, max_widths_major, lw=7, linestyle='-.', c='gray', label="Elastic HS Major Axis Profile", alpha=0.7)
        plt.scatter(elastic_depths, max_widths_major, s=200, edgecolors='black', zorder=10, c='gray', alpha=0.5)

    # plot viscoelastic models if requested
    if visc_models != None:
        max_widths = []
        max_widths_major = []

        for model, times in zip(visc_models, visc_times):
            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
            min_vel = np.infty
            FWHM = 0
            FWHM_major = 0
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                
                if min(vel_z) < min_vel:
                    min_vel = min(vel_z)

                    idx = np.argmin(np.abs(vel_z[x<0] - min_vel/2))
                    idx_major = np.argmin(np.abs(vel_z_major[x<0] - min_vel/2))

                    FWHM = abs(2*x[x<0][idx])
                    FWHM_major = abs(2*x[x<0][idx_major])

            max_widths.append((FWHM/1e3)/minor_norm)
            max_widths_major.append((FWHM_major/1e3)/major_norm)
            
        
        plt.plot(visc_depths, max_widths, lw=7, c='rosybrown', linestyle='dotted', label="Viscoelastic HS Minor Axis Profile", alpha=0.7)
        plt.scatter(visc_depths, max_widths, s=200, edgecolors='black', zorder=10, c='rosybrown', alpha=0.5)

        plt.plot(visc_depths, max_widths_major, lw=7, linestyle='-.', c='rosybrown', label="Viscoelastic HS Major Axis Profile", alpha=0.7)
        plt.scatter(visc_depths, max_widths_major, s=200, edgecolors='black', zorder=10, c='rosybrown', alpha=0.5)

    # plot parameters
    plt.xlabel(r"$d_s$ (km)", fontsize=30)
    if CR_norm == None:
        plt.ylabel("FWHM (km)", fontsize=30)
    else:
        plt.ylabel(r"FWHM$/r_{CR,x}$", fontsize=30)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=30)
    plt.grid()

    file_name = "/home/grantblock/Research/SMBPylith/Figures/depth_width_trend"
    plt.savefig(file_name, bbox_inches="tight")
    
    plt.show()

# function to plot trend of source size and FWHM
def plot_size_width_trend(models, times, CR_dims, add_model=None, add_label=None, elastic_models=None, visc_models=None, visc_times=None, CR_norm=1):


    # iterate through models
    minor_FWHM_list = []
    major_FWHM_list = []
    minor_CR_s_dist = []
    major_CR_s_dist = []
    for model in models:        

        # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
        max_vel = -np.infty
        FWHM = 0
        FWHM_major = 0
        for t in times:
            model_time = model.get_timesteps(t)
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                
            if max(abs(vel_z)) > max_vel:
                max_vel = max(abs(vel_z))

                idx = np.argmin(np.abs(np.abs(vel_z[x<0]) - max_vel/2))
                idx_major = np.argmin(np.abs(np.abs(vel_z_major[x<0]) - max_vel/2))

                FWHM = abs(2*x[x<0][idx])
                FWHM_major = abs(2*x[x<0][idx_major])

        minor_FWHM_list.append((FWHM/1e3)/CR_norm)
        major_FWHM_list.append((FWHM_major/1e3)/CR_norm)
        minor_CR_s_dist.append(CR_dims[0]-model.r_x)
        major_CR_s_dist.append(CR_dims[1]-model.r_y)


    # if there's an additional model to plot, calculate its FWHM and plot it
    if add_model != None:

        max_vel = -np.infty
        FWHM = 0
        FWHM_major = 0
        for t in times:
            model_time = add_model.get_timesteps(t)
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = add_model.get_data(model_time, theta=None)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = add_model.get_data(model_time, theta=np.pi/2.)
                
            if max(abs(vel_z)) > max_vel:
                max_vel = max(abs(vel_z))

                idx = np.argmin(np.abs(np.abs(vel_z[x<0]) - max_vel/2))
                idx_major = np.argmin(np.abs(np.abs(vel_z_major[x<0]) - max_vel/2))

                FWHM = abs(2*x[x<0][idx])
                FWHM_major = abs(2*x[x<0][idx_major])

        
        # plot additional model
        plt.scatter([CR_dims[0]-add_model.r_x], [(FWHM/1e3)/CR_norm], marker="*", s=500, c="green", label=add_label+" Minor Axis Profile", zorder=10)
        plt.scatter([CR_dims[1]-add_model.r_y], [(FWHM_major/1e3)/CR_norm], marker="*", s=500, c="purple", label=add_label+" Major Axis Profile", zorder=10)

    # If there are elastic models added, plot them
    if elastic_models != None:

        # iterate through models
        elastic_minor_FWHM_list = []
        elastic_major_FWHM_list = []
        elastic_minor_CR_s_dist = []
        elastic_major_CR_s_dist = []
        for model in elastic_models:        

            # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
            max_vel = -np.infty
            FWHM = 0
            FWHM_major = 0
            for t in times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                    
                if max(abs(vel_z)) > max_vel:
                    max_vel = max(abs(vel_z))

                    idx = np.argmin(np.abs(np.abs(vel_z[x<0]) - max_vel/2))
                    idx_major = np.argmin(np.abs(np.abs(vel_z_major[x<0]) - max_vel/2))

                    FWHM = abs(2*x[x<0][idx])
                    FWHM_major = abs(2*x[x<0][idx_major])

            elastic_minor_FWHM_list.append((FWHM/1e3)/CR_norm)
            elastic_major_FWHM_list.append((FWHM_major/1e3)/CR_norm)
            elastic_minor_CR_s_dist.append(CR_dims[0]-model.r_x)
            elastic_major_CR_s_dist.append(CR_dims[1]-model.r_y)

        plt.plot(elastic_minor_CR_s_dist, elastic_minor_FWHM_list, lw=7, c='gray', label="Elastic HS Minor Axis Profile")
        plt.scatter(elastic_minor_CR_s_dist, elastic_minor_FWHM_list, s=200, c='gray', zorder=10)

        plt.plot(elastic_major_CR_s_dist, elastic_major_FWHM_list, lw=7, linestyle='dashed', c='gray', label="Elastic HS Major Axis Profile")
        plt.scatter(elastic_major_CR_s_dist, elastic_major_FWHM_list, s=200, c='gray', zorder=10)

    # If there are visc models added, plot them
    if visc_models != None:

        # iterate through models
        visc_minor_FWHM_list = []
        visc_major_FWHM_list = []
        visc_minor_CR_s_dist = []
        visc_major_CR_s_dist = []
        for model in visc_models:        

            # loop through times to find the max magnitude vel and then the FWHM on minor axis and major axis profiles
            max_vel = -np.infty
            FWHM = 0
            FWHM_major = 0
            for t in visc_times:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_major, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2.)
                    
                if max(abs(vel_z)) > max_vel:
                    max_vel = max(abs(vel_z))

                    idx = np.argmin(np.abs(np.abs(vel_z[x<0]) - max_vel/2))
                    idx_major = np.argmin(np.abs(np.abs(vel_z_major[x<0]) - max_vel/2))

                    FWHM = abs(2*x[x<0][idx])
                    FWHM_major = abs(2*x[x<0][idx_major])

            visc_minor_FWHM_list.append((FWHM/1e3)/CR_norm)
            visc_major_FWHM_list.append((FWHM_major/1e3)/CR_norm)
            visc_minor_CR_s_dist.append(CR_dims[0]-model.r_x)
            visc_major_CR_s_dist.append(CR_dims[1]-model.r_y)

        plt.plot(visc_minor_CR_s_dist, np.asarray(visc_minor_FWHM_list)+0.2, lw=7, c='rosybrown', label="Viscoelastic HS Minor Axis Profile") # shift applied because bottom
                                                                                                                                # profile is not smooth so we have
                                                                                                                                # to use the top
        plt.scatter(visc_minor_CR_s_dist, np.asarray(visc_minor_FWHM_list)+0.2, s=200, c='rosybrown', zorder=10)

        plt.plot(visc_major_CR_s_dist, np.asarray(visc_major_FWHM_list)+0.42, lw=7, linestyle='dashed', c='rosybrown', label="Viscoelastic HS Major Axis Profile")
        plt.scatter(visc_major_CR_s_dist, np.asarray(visc_major_FWHM_list)+0.42, s=200, c='rosybrown', zorder=10)

    # plot
    plt.plot(minor_CR_s_dist, minor_FWHM_list, lw=7, c='black', label="Minor Axis Profile")
    plt.scatter(minor_CR_s_dist, minor_FWHM_list, s=200, c='black', zorder=10)

    plt.plot(major_CR_s_dist, major_FWHM_list, lw=7, linestyle='dashed', c='black', label="Major Axis Profile")
    plt.scatter(major_CR_s_dist, major_FWHM_list, s=200, c='black', zorder=10)

    # plot params
    plt.xlabel(r"$r_{CR}-r_s$ (km)", fontsize=30)
    if CR_norm == 1:
        plt.ylabel("FWHM (km)", fontsize=30)
    else:
        plt.ylabel(r"FWHM$/r_{CR,x}$", fontsize=30)


    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=30)
    plt.grid()

    file_name = "/home/grantblock/Research/SMBPylith/Figures/size_width_trend"
    plt.savefig(file_name, bbox_inches="tight")

    plt.show()

# Function to plot velocity vs. depth for different viscosities and fit curves to them
def plot_vel_depth_visc(model_lists, depth_lists, visc_list, times_list, pt_norm=True):

    alphas = []
    
    # iterate through each model list
    for models, depths, visc, times in zip(model_lists, depth_lists, visc_list, times_list):

        # for each model in the list, get the max magnitude vel
        # in the time span
        max_vels = []

        for model, times_model in zip(models, times):

            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to get the max vel
            # print(model.model.model_name, times)
            vel_list = []
            for t in times_model:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr*scaling_param)

            max_vels.append(-min(np.asarray(vel_list)))

        # find fitting parameters
        def base_function(d, V, alpha):

            return V/(d**alpha)
        
        param, param_cov = curve_fit(base_function, depths, max_vels)
        alphas.append(param[1])

        # plot 
        plt.plot(depths, max_vels, lw=7)
        plt.scatter(depths, max_vels, s=200, label=r"$\eta_{CR}$="+str(visc)+r" Pa s, $\alpha$="+str(round(param[1], 3)), edgecolors='black', zorder=10)


    # plot parameters
    plt.xlabel(r"$d_s$ (km)", fontsize=30)
    if pt_norm:
        plt.ylabel(r"$c_{vp}|V_z|$", fontsize=30)
    else:
        plt.ylabel(r"$|V_z|$ (mm/yr)", fontsize=30)

    plt.xticks(fontsize=25)
    plt.yscale('log')
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left',fontsize=30)
    plt.grid()

    file_name = "/home/grantblock/Research/SMBPylith/Figures/depth_vel_visc"
    plt.savefig(file_name, bbox_inches="tight")

    plt.show()

    # Plot alpha values with viscosity
    plt.plot(np.asarray(visc_list)/1e18, alphas, lw=7, c='black')
    plt.scatter(np.asarray(visc_list)/1e18, alphas, s=200, c='black', zorder=10)
    
    plt.xlabel(r"$\eta_{CR}/1E18$ (Pa s)", fontsize=30)
    plt.ylabel(r"$\alpha$", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.show()

# Function to plot velocity vs. viscosity for different depths
def plot_vel_visc(model_lists, depth_list, visc_list, times_list, pt_norm=True):

    # iterate through model lists
    for models, depth, times in zip(model_lists, depth_list, times_list):

        # for each model in the list, get the max magnitude vel
        # in the time span
        max_vels = []

        for model, times_model in zip(models, times):

            # set scaling
            if pt_norm:
                d_uc=10e3
                scaling_param = model.model.P0/(model.model.Delta_P*d_uc)*1e-3 # in yr/mm
            else:
                scaling_param = 1

            # loop through times to get the max vel
            # print(model.model.model_name, times)
            vel_list = []
            for t in times_model:
                model_time = model.get_timesteps(t)
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)
                vel_list.append(vel_z[np.abs(x).argmin()]*model.ms_to_mmyr*scaling_param)

            max_vels.append(-min(np.asarray(vel_list)))

        # plot
        plt.plot(np.asarray(visc_list)/1e18, max_vels, lw=7)
        plt.scatter(np.asarray(visc_list)/1e18, max_vels, s=200, label=r"$d_s$="+str(depth)+" km", edgecolors='black', zorder=10)

    # plot parameters
    plt.xlabel(r"$\eta_{CR}/1E18$ Pa s", fontsize=30)
    if pt_norm:
        plt.ylabel(r"$c_{vp}|V_z|$", fontsize=30)
    else:
        plt.ylabel(r"$|V_z|$ (mm/yr)", fontsize=30)

    plt.xticks(fontsize=25)
    plt.yscale('log')
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left',fontsize=30)
    plt.grid()

    file_name = "/home/grantblock/Research/SMBPylith/Figures/vel_visc"
    plt.savefig(file_name, bbox_inches="tight")

    plt.show()    


#-------------------------------------------------------------------------#

# Model declarations
Yellowstone_Run2 = AnalyzeModel.Analysis("Yellowstone_Run2", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run2_run2 = AnalyzeModel.Analysis("Yellowstone_Run2_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run3 = AnalyzeModel.Analysis("Yellowstone_Run3", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run3_run2 = AnalyzeModel.Analysis("Yellowstone_Run3_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run4 = AnalyzeModel.Analysis("Yellowstone_Run4", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run4_run2 = AnalyzeModel.Analysis("Yellowstone_Run4_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run5 = AnalyzeModel.Analysis("Yellowstone_Run5", r_x=0.25*13, r_y=0.25*27.5)
Yellowstone_Run5_run2 = AnalyzeModel.Analysis("Yellowstone_Run5_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run6 = AnalyzeModel.Analysis("Yellowstone_Run6", r_x=0.25*13, r_y=0.25*27.5)
Yellowstone_Run6_run2 = AnalyzeModel.Analysis("Yellowstone_Run6_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run7 = AnalyzeModel.Analysis("Yellowstone_Run7", r_x=0.25*13, r_y=0.25*27.5)
Yellowstone_Run7_run2 = AnalyzeModel.Analysis("Yellowstone_Run7_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run8_run2 = AnalyzeModel.Analysis("Yellowstone_Run8_run2", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run8_run3 = AnalyzeModel.Analysis("Yellowstone_Run8_run3", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run8_run4 = AnalyzeModel.Analysis("Yellowstone_Run8_run4", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
# Yellowstone_Run118 = AnalyzeModel.Analysis("Yellowstone_Run118", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
# Yellowstone_Run119 = AnalyzeModel.Analysis("Yellowstone_Run119", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
# Yellowstone_Run120 = AnalyzeModel.Analysis("Yellowstone_Run120", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
# Yellowstone_Run124 = AnalyzeModel.Analysis("Yellowstone_Run124", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
# Yellowstone_Run127 = AnalyzeModel.Analysis("Yellowstone_Run127", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run219 = AnalyzeModel.Analysis("Yellowstone_Run219", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run219_run2 = AnalyzeModel.Analysis("Yellowstone_Run219_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run219_run3 = AnalyzeModel.Analysis("Yellowstone_Run219_run3", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run219_run4 = AnalyzeModel.Analysis("Yellowstone_Run219_run4", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run219_run5 = AnalyzeModel.Analysis("Yellowstone_Run219_run5", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run220 = AnalyzeModel.Analysis("Yellowstone_Run220", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run220_run2 = AnalyzeModel.Analysis("Yellowstone_Run220_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run220_run3 = AnalyzeModel.Analysis("Yellowstone_Run220_run3", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run220_run4 = AnalyzeModel.Analysis("Yellowstone_Run220_run4", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run220_run5 = AnalyzeModel.Analysis("Yellowstone_Run220_run5", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)

# visc varying runs
Yellowstone_Run221 = AnalyzeModel.Analysis("Yellowstone_Run221", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run221_run2 = AnalyzeModel.Analysis("Yellowstone_Run221_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run222 = AnalyzeModel.Analysis("Yellowstone_Run222", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run222_run2 = AnalyzeModel.Analysis("Yellowstone_Run222_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run222_run3 = AnalyzeModel.Analysis("Yellowstone_Run222_run3", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run223 = AnalyzeModel.Analysis("Yellowstone_Run223", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run223_run2 = AnalyzeModel.Analysis("Yellowstone_Run223_run2", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run224 = AnalyzeModel.Analysis("Yellowstone_Run224", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run224_run2 = AnalyzeModel.Analysis("Yellowstone_Run224_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run225 = AnalyzeModel.Analysis("Yellowstone_Run225", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run225_run2 = AnalyzeModel.Analysis("Yellowstone_Run225_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run225_run3 = AnalyzeModel.Analysis("Yellowstone_Run225_run3", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run226 = AnalyzeModel.Analysis("Yellowstone_Run226", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run226_run2 = AnalyzeModel.Analysis("Yellowstone_Run226_run2", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run227 = AnalyzeModel.Analysis("Yellowstone_Run227", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run227_run2 = AnalyzeModel.Analysis("Yellowstone_Run227_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run228 = AnalyzeModel.Analysis("Yellowstone_Run228", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run228_run2 = AnalyzeModel.Analysis("Yellowstone_Run228_run2", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run228_run3 = AnalyzeModel.Analysis("Yellowstone_Run228_run3", r_x=0.5*13, r_y=0.5*27.5, mesh_width=100e3)
Yellowstone_Run229 = AnalyzeModel.Analysis("Yellowstone_Run229", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)
Yellowstone_Run229_run2 = AnalyzeModel.Analysis("Yellowstone_Run229_run2", r_x=0.8*13, r_y=0.8*27.5, mesh_width=100e3)

# nested runs
Yellowstone_Run230 = AnalyzeModel.Analysis("Yellowstone_Run230", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run230_run2 = AnalyzeModel.Analysis("Yellowstone_Run230_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run230_run3 = AnalyzeModel.Analysis("Yellowstone_Run230_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run231 = AnalyzeModel.Analysis("Yellowstone_Run231", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run231_run2 = AnalyzeModel.Analysis("Yellowstone_Run231_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run231_run3 = AnalyzeModel.Analysis("Yellowstone_Run231_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run232 = AnalyzeModel.Analysis("Yellowstone_Run232", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run232_run2 = AnalyzeModel.Analysis("Yellowstone_Run232_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run232_run3 = AnalyzeModel.Analysis("Yellowstone_Run232_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run233 = AnalyzeModel.Analysis("Yellowstone_Run233", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run233_run2 = AnalyzeModel.Analysis("Yellowstone_Run233_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run233_run3 = AnalyzeModel.Analysis("Yellowstone_Run233_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)

# Varying timedb
Yellowstone_Run234 = AnalyzeModel.Analysis("Yellowstone_Run234", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run234_run2 = AnalyzeModel.Analysis("Yellowstone_Run234_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run234_run3 = AnalyzeModel.Analysis("Yellowstone_Run234_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run235 = AnalyzeModel.Analysis("Yellowstone_Run235", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run235_run2 = AnalyzeModel.Analysis("Yellowstone_Run235_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run235_run3 = AnalyzeModel.Analysis("Yellowstone_Run235_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)

# Varying T
Yellowstone_Run236 = AnalyzeModel.Analysis("Yellowstone_Run236", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run236_run2 = AnalyzeModel.Analysis("Yellowstone_Run236_run2", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)
Yellowstone_Run236_run3 = AnalyzeModel.Analysis("Yellowstone_Run236_run3", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)

# 10 cycles
Yellowstone_Run237 = AnalyzeModel.Analysis("Yellowstone_Run237", r_x=0.25*13, r_y=0.25*27.5, mesh_width=100e3)




# Function calls

# Function calls for Figures 3, 4, dp/dt scaling 
# dpdt_scaling_plot([Yellowstone_Run118, Yellowstone_Run119, Yellowstone_Run120], 510)
# plot_pressure_func(Yellowstone_Run120, "/home/grantblock/Research/Yellowstone/timedb/run120.timedb")

# Function call used in Figure 3, scaling relations
# visc_scaling_plot([Yellowstone_Run120, Yellowstone_Run124, Yellowstone_Run127], [1.2616e17, 6.308e16, 1.2616e16], [10e9, 10e9, 10e9], [5e3, 5e3, 5e3], 510)

# Function calls used in Figure 5, depth variations
# plot_profile_time(Yellowstone_Run221, [511.3, 515], [r"$t=t_{rise}$", r"$t=T$"])
# Yellowstone_Run220_run3.plot_points_over_time(np.arange(4990, 5035, 1), [(0, 0), (13.0, 0)], save=False, norm=False,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run220_run3.timedb")
# plot_groundsurf_vel(Yellowstone_Run3_run2, 3011.3, vel_lim=[-30, 20], CR=[13, 27.5], x_lim=[-30, 30], y_lim=[-35, 35])
# plot_groundsurf_vel(Yellowstone_Run3_run2, 3015, vel_lim=[-30, 20], CR=[13, 27.5], x_lim=[-30, 30], y_lim=[-35, 35])

# Function calls for Figure 7, vel and width trends with depth
# plot_depth_width_trend([[Yellowstone_Run6_run2, Yellowstone_Run5_run2, Yellowstone_Run7_run2], 
#                         [Yellowstone_Run3_run2, Yellowstone_Run2_run2, Yellowstone_Run4_run2],
#                         [Yellowstone_Run8_run3, Yellowstone_Run8_run2, Yellowstone_Run8_run4]], 
#                        [[15, 6.5, 4], [15, 6.5, 4], [15, 6.5, 4]], [np.arange(3000, 3030.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5)], 
#                        ["Source dims = 25% CR dims", "Source dims = 50% CR dims", "Source dims = 80% CR dims"], depth_norm=False,
#                        elastic_models=[Yellowstone_Run219_run2, Yellowstone_Run219, Yellowstone_Run219_run3], 
#                        elastic_depths=[15, 6.5, 4],
#                        elastic_times=[np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5)],
#                        elastic_label="Elastic half space, Source dims = 50% CR dims",
#                        visc_models=[Yellowstone_Run220_run2, Yellowstone_Run220, Yellowstone_Run220_run3],
#                        visc_depths=[15, 6.5, 4],
#                        visc_times=[np.arange(5000, 5033.5, 0.5), np.arange(5000, 5033.5, 0.5), np.arange(5000, 5033.5, 0.5)],
#                        visc_label="Viscoelastic half space, Source dimes = 50% CR dims",
#                        CR_norm=[13, 13])

# plot_size_width_trend([Yellowstone_Run5_run2, Yellowstone_Run2_run2, Yellowstone_Run8_run2], np.arange(500, 530.5, 0.5), [13, 27.5],
#                       add_model=Yellowstone_Run2, add_label="Ellipsoidal CR,", 
#                       elastic_models=[Yellowstone_Run219_run4, Yellowstone_Run219, Yellowstone_Run219_run5],
#                       visc_models=[Yellowstone_Run220_run4, Yellowstone_Run220, Yellowstone_Run220_run5],
#                       visc_times=np.arange(5000, 5033.5, 0.5), CR_norm=13)

# Function calls for figure 6, profiles and 2D groundurface varying width
# plot_profile_time(Yellowstone_Run2_run2, [511.3, 515], [r"$t=t_{rise}$", r"$t=T$"], plot_source_lines=True, CR_lines=[-13, 13])
# plot_groundsurf_vel(Yellowstone_Run2, 515, vel_lim=[-20, 15], CR=[13, 27.5], x_lim=[-30, 30], y_lim=[-35, 35])

# Function calls for Fig 8, first CC fig
# Yellowstone_Run8_run4.plot_cc_surface_time(np.arange(500, 516, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[0.5, 1])
# Yellowstone_Run7_run2.get_cc_center_point(np.arange(500, 516, 1), [0, -25], debug=True)
# Yellowstone_Run5_run2.plot_ratio_surface_time(np.arange(500, 516, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5])

# Function calls for Fig 9
# plot_vel_depth_visc([[Yellowstone_Run4_run2, Yellowstone_Run2_run2, Yellowstone_Run3_run2],
#                      [Yellowstone_Run222, Yellowstone_Run222_run2, Yellowstone_Run222_run3],
#                      [Yellowstone_Run225, Yellowstone_Run225_run2, Yellowstone_Run225_run3],
#                      [Yellowstone_Run228, Yellowstone_Run228_run2, Yellowstone_Run228_run3]], 
#                       [[4.0, 6.5, 15.0], [4.0, 6.5, 15.0], [4.0, 6.5, 15.0], [4.0, 6.5, 15.0]], 
#                       [1.2616e18, 6.308e17, 1.2616e17, 6.308e16], 
#                       [[np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(3000, 3030.5, 0.5)],
#                        [np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(2000, 2030.5, 0.5)],
#                        [np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(1000, 1030.5, 0.5)],
#                        [np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5)]])

# plot_vel_visc([[Yellowstone_Run4_run2, Yellowstone_Run222, Yellowstone_Run225, Yellowstone_Run228],
#                [Yellowstone_Run2_run2, Yellowstone_Run222_run2, Yellowstone_Run225_run2, Yellowstone_Run228_run2],
#                [Yellowstone_Run3_run2, Yellowstone_Run222_run3, Yellowstone_Run225_run3, Yellowstone_Run228_run3]], 
#                [4.0, 6.5, 15.0], [1.2616e18, 6.308e17, 1.2616e17, 6.308e16], 
#                [[np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5)],
#                 [np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5), np.arange(500, 530.5, 0.5)],
#                 [np.arange(3000, 3030.5, 0.5), np.arange(2000, 2030.5, 0.5), np.arange(1000, 1030.5, 0.5), np.arange(500, 530.5, 0.5)]])

# Yellowstone_Run229.plot_cc_surface_time(np.arange(500, 516, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[-1, 1])
# Yellowstone_Run221.plot_ratio_surface_time(np.arange(500, 516, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], ratio_lim=[10, 100])
# Yellowstone_Run224.get_ratio_center_point(np.arange(500, 516, 1), (0, -5), debug=True)

# plot_profile_time(Yellowstone_Run224, [515], [r"$t=T$"], CR_lines=[-27.5, 27.5])
# plot_groundsurf_vel(Yellowstone_Run224, 515, vel_lim=[-10, 5], CR=[13, 27.5], x_lim=[-30, 30], y_lim=[-35, 35])
# plot_groundsurf_vel(Yellowstone_Run230_run3, 515, vel_lim=[-10, 5], CR=[13, 27.5], CR_inner=[0.6*13, 0.6*27.5], x_lim=[-30, 30], y_lim=[-35, 35])

# Yellowstone_Run230_run3.plot_cc_surface_time(np.arange(500, 516, 1), (13, 27.5), inner_CR=[0.6*13, 0.6*27.5], save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[-1, 1])
# Yellowstone_Run230_run3.plot_ratio_surface_time(np.arange(500, 516, 1), (13, 27.5),  inner_CR=[0.6*13, 0.6*27.5], save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], ratio_lim=[10, 100])

# Yellowstone_Run235_run3.plot_points_over_time(np.arange(498, 536, 1), [(0, 0), (-0.25*6.5, 0), (-0.5*6.5, 0), (-0.75*6.5, 0), (-6.5, 0)], save=True, norm=False, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run235_run3.timedb")
# Yellowstone_Run235_run3.plot_points_over_time(np.arange(498, 536, 1), [(0, 0), (0, -0.25*27.5), (0, -0.5*27.5), (0, -0.75*27.5), (0, -27.5)], save=True, norm=True, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run235_run3.timedb")

# Yellowstone_Run235_run3.plot_cc_surface_time(np.arange(515, 531, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[-1, 1])
# Yellowstone_Run235_run3.plot_ratio_surface_time(np.arange(515, 531, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], ratio_lim=[10, 100])

# Yellowstone_Run236_run3.plot_points_over_time(np.arange(498, 605, 1), [(0, 0), (-0.25*6.5, 0), (-0.5*6.5, 0), (-0.75*6.5, 0), (-6.5, 0)], save=True, norm=False, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run236_run3.timedb")
# Yellowstone_Run236_run3.plot_points_over_time(np.arange(498, 605, 1), [(0, 0), (0, -0.25*27.5), (0, -0.5*27.5), (0, -0.75*27.5), (0, -27.5)], save=True, norm=True, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run236_run3.timedb")

# Yellowstone_Run236_run3.plot_cc_surface_time(np.arange(565, 586, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[-1, 1])
# Yellowstone_Run236_run3.plot_ratio_surface_time(np.arange(550, 601.5, 1.5), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], ratio_lim=[10, 100])


# Yellowstone_Run237.plot_points_over_time(np.arange(498, 655, 1), [(0, 0), (-0.25*6.5, 0), (-0.5*6.5, 0), (-0.75*6.5, 0), (-6.5, 0)], save=True, norm=False, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run237.timedb")
# Yellowstone_Run237.plot_points_over_time(np.arange(498, 655, 1), [(0, 0), (0, -0.25*27.5), (0, -0.5*27.5), (0, -0.75*27.5), (0, -27.5)], save=True, norm=True, vel_scale=True,
#                                         pressure_func_file="/home/grantblock/Research/Yellowstone/timedb/run237.timedb")

# Yellowstone_Run237.plot_cc_surface_time(np.arange(635, 651, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], CC_lim=[-1, 1])
Yellowstone_Run237.plot_ratio_surface_time(np.arange(635, 651, 1), (13, 27.5), save=True, x_lim=[-13, 13], y_lim=[-27.5, 27.5], ratio_lim=[10, 100])


