import numpy as np
import ModelStorage
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib
import os
from scipy.signal import find_peaks
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.optimize import curve_fit
import pandas as pd
import sys
import seaborn as sns

# Class to read in and analyze hdf5 output from pylith
# it will find the derived parameters then write them to the appropriate place
# in the master csv
class Analysis:
    def __init__(self, model_name, r_x=25, r_y=None, x_off=0.0, y_off=0.0, mesh_width=150e3):
        self.model_name = model_name
        self.model = ModelStorage.Model(model_name)
        self.ms_to_mmyr = 3.154e+10 #m/s to mm/yr velocity conversion
        self.shoulder_ratio = 1#1.6 #The shoulder (for finding sombreros) is defined to be 1.6*r_source from the center
        self.r_x = r_x #km. Radius of source in x direction. Set to 25 as a default for most models. Needed to calculate sombreros. 
        self.mesh_width = mesh_width
        if r_y == None: #km. Radius of source in y direction. Equal to r_x in case of cylindrically symmetric sources.
            self.r_y = r_x
        else:
            self.r_y = r_y 
        
        self.x_off = x_off*1e3 #x and y offset to interpolate with. Should be passed in in km but needed in m for read_hdf5().
        self.y_off = y_off*1e3

        #default plot formatting parameters
        self.lw = 4
        self.ms = 75
        self.label_fontsize=15
        self.axes_fontsize=15
        self.legend_fontsize=15
        self.title_fontsize=20
        self.tick_fontsize=15

        self.model_data = {}    #initiate as empty dict which will will store
                                #model data at a time step so the hdf5 will only 
                                #need to be read once per time step for a given
                                #model object.
        
        self.source_velocity = {} #a dictionary to store the averaged velocity over the full source at a given time
        
        # get timestep list
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5"
        with h5py.File(path, "r") as f:
            self.times = f['time'][:,0,0]



    #read hdf5 and store model data
    def read_hdf5(self, time_step, theta=None, x_off=0.0, y_off=0.0):

        #check to see if time step is already in the dictionary
        if (time_step, theta, x_off, y_off) in self.model_data:
            print("Time step has already been read from hdf5.")
            return
    
        #get the file path
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 
    
        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            # print(f['time'][:,0,0])
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            disp_x = displacements[time_step][:][:,0]
            disp_y = displacements[time_step][:][:,1]
            disp_z = displacements[time_step][:][:,2]
            
            vel_x = velocities[time_step][:][:,0]
            vel_y = velocities[time_step][:][:,1]
            vel_z = velocities[time_step][:][:,2]

            #get radial components
            x[x==0] = 0.0001
            angle = np.arctan(y/x)

            disp_r = disp_x*np.cos(angle)+disp_y*np.sin(angle)
            disp_theta = -disp_x*np.sin(angle)+disp_y*np.cos(angle)
        
            vel_r = vel_x*np.cos(angle)+vel_y*np.sin(angle)
            vel_theta = -vel_x*np.sin(angle)+vel_y*np.cos(angle)
            
            
            
            #interpolate data 
            #get points to interpolate
            x_i = np.linspace(-self.mesh_width, self.mesh_width, 11615)

            if theta == None:
                y_i = np.zeros(len(x_i))
            else:
                x_i_temp = x_i
                x_i = np.cos(theta)*x_i_temp 
                y_i = np.sin(theta)*x_i_temp 
                
            x_i += x_off
            y_i += y_off
            
            #interpolate
            disp_x_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_x), np.asarray([x_i, y_i]).T, method='cubic')
            disp_y_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_y), np.asarray([x_i, y_i]).T, method='cubic')
            disp_z_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_z), np.asarray([x_i, y_i]).T, method='cubic')
            
            vel_x_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_x), np.asarray([x_i, y_i]).T, method='cubic')
            vel_y_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_y), np.asarray([x_i, y_i]).T, method='cubic')
            vel_z_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_z), np.asarray([x_i, y_i]).T, method='cubic')

            disp_r_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_r), np.asarray([x_i, y_i]).T, method='cubic')
            disp_theta_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_theta), np.asarray([x_i, y_i]).T, method='cubic')
        
            vel_r_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_r), np.asarray([x_i, y_i]).T, method='cubic')
            vel_theta_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_theta), np.asarray([x_i, y_i]).T, method='cubic')

            # add all values to dictionary at the time step
            self.model_data[(time_step, theta, x_off, y_off)] = (x_i, disp_x_interp, disp_y_interp, disp_z_interp, disp_r_interp, disp_theta_interp, 
                                          vel_x_interp, vel_y_interp, vel_z_interp, vel_r_interp, vel_theta_interp)

    def get_point(self, time_step, x_in, y_in):

        #get the file path
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 
    
        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            # print(f['time'][:,0,0])
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            vel_z = velocities[time_step][:][:,2]

            mesh_width = 150e3

            grid_x, grid_y = np.mgrid[-mesh_width:mesh_width:1000j, -mesh_width:mesh_width:1000j]

            z = interpolate.griddata((x/1e3,y/1e3), vel_z, (-grid_x/1e3, -grid_y/1e3), method='cubic').T

            space_arr = np.linspace(-mesh_width, mesh_width, 1000)

            return z[np.argmin(np.abs(x_in-space_arr))][np.argmin(np.abs(y_in-space_arr))]

    # gets the length of a mesh element closest to a point defined
    # by an r and theta. 
    def get_mesh_size(self, r, theta):

        # first read in mesh (just get x and y points)
        if self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 
    
        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            
            points = group_geometry['vertices'] #shape: point_num, xyz

            x_mesh_list = points[:][:,0]
            y_mesh_list = points[:][:,1]



            # convert the passed in r and theta to cartesian
            x_model = r*np.cos(theta)
            y_model = r*np.sin(theta)

            # find the closest mesh point to the model point
            # mesh_idx = np.argmin(np.sqrt((x_mesh_list-x_model)**2+(y_mesh_list-y_model)**2))

            # x_mesh = x_mesh_list[mesh_idx]
            # y_mesh = y_mesh_list[mesh_idx]
            
            #x_mesh, y_mesh= get_nearest_point(x_model, y_model, x_mesh_list, y_mesh_list, no_same_point=False)

            distances = np.sqrt((x_mesh_list-x_model)**2+(y_mesh_list-y_model)**2)
            distances_sort = np.sort(distances)
            distance1_idx = np.where(distances==distances_sort[0])
            distance2_idx = np.where(distances==distances_sort[1])

            x_mesh = x_mesh_list[distance1_idx]
            y_mesh = y_mesh_list[distance1_idx]

            x_mesh_nearest = x_mesh_list[distance2_idx]
            y_mesh_nearest = y_mesh_list[distance2_idx]
            

            # x_mesh1 = x_mesh_list[distance1_idx]
            # y_mesh1 = y_mesh_list[distance1_idx]
            # x_mesh2 = x_mesh_list[distance2_idx]
            # y_mesh2 = y_mesh_list[distance2_idx]

            # print(distance1_idx, distance2_idx)
            # print(x_model, y_model, x_mesh1, y_mesh1, x_mesh2, y_mesh2)

            # find the point on the mesh nearest to the one we found
            # x_mesh_list_del = np.delete(x_mesh_list, mesh_idx)
            # y_mesh_list_del = np.delete(y_mesh_list, mesh_idx)
            # nearest_mesh_idx = np.argmin(np.sqrt((x_mesh_list_del-x_mesh)**2+(y_mesh_list_del-y_mesh)**2))
            # x_mesh_nearest = x_mesh_list[nearest_mesh_idx]
            # y_mesh_nearest = y_mesh_list[nearest_mesh_idx]

            #x_mesh_nearest, y_mesh_nearest = get_nearest_point(x_mesh, y_mesh, x_mesh_list, y_mesh_list, no_same_point=True)

            # calculate and return the distance between the two closest mesh points
            distance = np.sqrt((x_mesh_nearest-x_mesh)**2+(y_mesh_nearest-y_mesh)**2)[0]

            print(x_model, y_model, x_mesh, y_mesh, x_mesh_nearest, y_mesh_nearest, distance)
            plt.scatter(x_mesh_list, y_mesh_list)
            plt.scatter(x_model, y_model, c="red")
            plt.scatter(x_mesh, y_mesh, c="green")
            plt.scatter(x_mesh_nearest, y_mesh_nearest, c="purple")
            plt.show()
            closest_point_rad = np.sqrt(x_mesh**2+y_mesh**2)
            return distance

    # a helper function to get displacements and velocities either from
    # the dictionary or read the hdf5 file then get the data from the dict
    def get_data(self, model_time, theta=None):
        if (model_time, theta, self.x_off, self.y_off) not in self.model_data:
            self.read_hdf5(model_time, theta=theta, x_off=self.x_off, y_off=self.y_off)
        return self.model_data[(model_time, theta, self.x_off, self.y_off)]

    def get_data_no_shift(self, model_time, theta=None):
        if (model_time, theta, 0, 0) not in self.model_data:
            self.read_hdf5(model_time, theta=theta, x_off=0, y_off=0)
        return self.model_data[(model_time, theta, 0,0)]

    
    # a function to convert times in years to timesteps for the h5 file. 
    # times: a time in years
    # dt: can be a float or a tuple of floats. 
    # if its a tuple: dt[0] = spinup dt dt[1] = other dt
    def get_timesteps(self, time):
        time_idx = (np.abs(time*3.154e+7 - self.times)).argmin() #get time index
        return time_idx
        # if type(self.model.output_dt) == list:
        #     if time <= self.model.spinup_time:
        #         return int(time/self.model.output_dt[0])
        #     else:
        #         return int(self.model.spinup_time/self.model.output_dt[0])+int((time-self.model.spinup_time)/self.model.output_dt[1])
        # else:
        #     return int(time/self.model.output_dt)
        

    #save data to a csv
    def save_csv(self, column_list, column_names, path):
        f = open(path, "w")
        for i in range(len(column_names)):
            if i == len(column_names)-1:
                f.write(column_names[i]+"\n")
            else:
                f.write(column_names[i]+",")
        for i in range(len(column_list[0])):
            for j in range(len(column_list)):
                if i >= len(column_list[j]):
                    if j == len(column_list)-1:
                        f.write("\n")
                else:
                    if j == len(column_list)-1:
                        f.write(str(column_list[j][i])+"\n")
                    else:
                        f.write(str(column_list[j][i])+",")

    # get a list of points enclosing the sombrero at its minimum for a given time step
    def get_som_width_contour(self, time_step):

        mesh_width = 150e3 #width of mesh from 0 to edge
        model_time = self.get_timesteps(time_step)

        # get a profile at every angle from the center
        angle_list = np.arange(0, 2*np.pi+0.1, 0.1)
        x_min_list = []
        y_min_list = []
        for theta in angle_list:

            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            if theta == 0:
                x_0 = x

            # find minimum point at x < 0 (as we rotate around all angles this will cover the full sombrero)
            prof_vel_neg = list(vel_z[x_0<0])
            neg_idx = prof_vel_neg.index(min(prof_vel_neg))

            x_i_temp = np.linspace(-mesh_width, mesh_width, len(x_0))    
            x_i = np.cos(theta)*x_i_temp 
            y_i = np.sin(theta)*x_i_temp 

            neg_min_x = x_i[neg_idx]
            neg_min_y = y_i[neg_idx]

            x_min_list.append(neg_min_x)
            y_min_list.append(neg_min_y)

        return x_min_list, y_min_list
        

    # plot vz profiles over a given set of time steps
    def plot_profiles_time(self, time_steps, theta=None, save=False, save_name=None, lim=None, shift_time=None,use_color_list=False):

        # get x_0 for plotting
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # set up plot
        plt.xlabel("Distance From Center of Source (km)", fontsize=20)
        plt.ylabel("Velocity (mm/yr)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()

        color_list=['lightpink', 'mediumvioletred', 'darkmagenta', 'royalblue', 'midnightblue']

        # iterate over time steps and plot
        list_itr=0
        for t in time_steps:

            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            x = x_0

            if shift_time == None:
                if use_color_list:
                    plt.plot(x_0/1e3, vel_z*self.ms_to_mmyr, linewidth=self.lw, color=color_list[list_itr], label="t="+str(round(t, 2))+" yrs")
                else:
                    plt.plot(x_0/1e3, vel_z*self.ms_to_mmyr, linewidth=self.lw, label="t="+str(round(t, 2))+" yrs")

            else:
                if use_color_list:
                    plt.plot(x_0/1e3, vel_z*self.ms_to_mmyr, linewidth=5, color=color_list[list_itr], label="t="+str(round(t-500+shift_time, 2)))
                else:
                    plt.plot(x_0/1e3, vel_z*self.ms_to_mmyr, linewidth=5, label="t="+str(round(t-500+shift_time, 2)))


            list_itr+=1
                
        
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=20)
        if lim != None:
            plt.xlim(lim)
        # save plot 
        if save:
            if save_name:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_time_"
                file_name += self.model_name.replace("/", "_") + save_name + ".png"
                plt.savefig(file_name, bbox_inches="tight")
            else:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_time_"
                file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
                plt.savefig(file_name, bbox_inches="tight")

        plt.show()

    def plot_profiles_mean_time(self, mean_times, theta=None, lim=None, shift_time=None, plot_points=None, station_locs=None):

        # get x_0 for plotting
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # set up plot
        fig, ax = plt.subplots(1, 1, figsize=(17, 6))
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("Distance From Center of Source (km)", fontsize=50)
        ax.set_ylabel("Velocity (mm/yr)", fontsize=50)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        # plt.yscale('symlog')
        ax.grid()


        profile_list = []
        for t in mean_times:
            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)

            profile_list.append(vel_z*self.ms_to_mmyr)
        
        mean_profile = np.mean(np.asarray(profile_list), axis=0)

        if theta == np.pi/2.:
            profile_legend = "YY' at <t>="+str(round(np.mean(np.asarray(mean_times))-500+shift_time, 2))
        else:
            profile_legend = "XX' at <t>="+str(round(np.mean(np.asarray(mean_times))-500+shift_time, 2))

        # plot

        ax.plot(x_0/1e3, mean_profile, linewidth=12, label=profile_legend, c='black')

        # if there are points to plot, iterate and plot them
        if plot_points != None:

            itr = 0
            for point in plot_points:
                # read GPS station
                df = pd.read_csv(point)
                GPS_times = df['time[yrs]'].values
                GPS_vel = df['velocity[m/yr]'].values

                time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)
                GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3

                if len(GPS_vel[time_cut]) > 0:
                    # print(point, GPS_mean_vel, np.asarray([[min(GPS_vel[time_cut]*1e3)], [max(GPS_vel[time_cut]*1e3)]]))
                
                    # plot point mean and range
                    if point in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                        # station_color = 'indigo'
                        station_color = 'black'
                        station_symbol = 's'
                    else:
                        # station_color = 'teal'
                        station_color = 'black'
                        station_symbol = '^'
                    if theta == np.pi/2.:
                        ax.errorbar(station_locs[itr][1], GPS_mean_vel, yerr=np.asarray([[GPS_mean_vel-min(GPS_vel[time_cut]*1e3)], [max(GPS_vel[time_cut]*1e3)-GPS_mean_vel]]), 
                                    fmt=station_symbol, c=station_color,  markersize='40', markeredgecolor='black', elinewidth=7)
                    else:
                        ax.errorbar(station_locs[itr][0], GPS_mean_vel, yerr=np.asarray([[GPS_mean_vel-min(GPS_vel[time_cut]*1e3)], [max(GPS_vel[time_cut]*1e3)-GPS_mean_vel]]), 
                                    fmt=station_symbol, c=station_color,  markersize='40', markeredgecolor='black', elinewidth=7)
                else:
                    print("station: ", point, " not included")
                itr+=1  

        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=20)
        # plt.legend(fontsize=30)
        if lim != None:
            ax.set_xlim(lim)

        file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_mean_time_"
        file_name += self.model_name.replace("/", "_") + ".png"
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
        
        plt.show()
    
    # Plot the model data residuals points along the model profile
    def plot_profile_res_mean_time(self, mean_times, theta=None, lim=None, shift_time=None, plot_points=None, station_locs=None):

        # get x_0 for plotting
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # set up plot
        fig, ax = plt.subplots(1, 1, figsize=(17, 6))
        ax.set_xlabel("Distance From Center of Source (km)", fontsize=40)
        ax.set_ylabel("Residual (mm/yr)", fontsize=40)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        ax.grid()
        

        # get mean time profile
        profile_list = []
        for t in mean_times:
            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)

            profile_list.append(vel_z*self.ms_to_mmyr)
        
        mean_profile = np.mean(np.asarray(profile_list), axis=0)

        # Get GPS station points
        itr = 0
        for point in plot_points:
            # read GPS station
            df = pd.read_csv(point)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values
            
            time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)
            GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3

            if len(GPS_vel[time_cut]) > 0:
                if point in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                    station_color = 'indigo'
                else:
                    station_color = 'teal'

                # Get the profile point correlating to the (projected) station location
                if theta == np.pi/2.:
                    loc = station_locs[itr][1]  
                else:
                    loc = station_locs[itr][0]
                    
                point_idx = (np.abs(x_0/1e3 - loc)).argmin()
                print(point, GPS_mean_vel, mean_profile[point_idx])
                res = GPS_mean_vel - mean_profile[point_idx]

                ax.scatter(loc, res, marker='s', c=station_color, s=500, edgecolors='black')
                
            else:
                print("station: ", point, " not included")
            itr+=1

        if lim != None:
            ax.set_xlim(lim)

        file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_mean_time_res"
        file_name += self.model_name.replace("/", "_") + ".png"
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
        
        plt.show()    


    # Plot the model data residuals points along the model profile
    def plot_profile_res_mean_time_new(self, mean_times, shift_time=None, stations=None, station_locs=None): 

        # get base profile
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # get mean model velocities at each point

        station_names = []
        loc = 0
        max_val = -np.infty
        min_val = np.infty
        for point, station in zip(station_locs, stations):
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)

            # get mean  modelvelocity
            model_vel_list = []
            for t in mean_times:
                model_time = self.get_timesteps(t)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
                point_idx = (np.abs(x_0 - r)).argmin() #get point index
                model_vel_list.append(vel_z[point_idx]*self.ms_to_mmyr)
            mean_model_vel = np.mean(np.asarray(model_vel_list), axis=0)

            # get mean station velocity
            df = pd.read_csv(station)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values
            
            time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)

            # if the station has data in the time cut, append to all of the lists
            if len(GPS_vel[time_cut]) > 0:

                if station in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                    station_color = 'black'
                    shape = 's'
                else:
                    # station_color = 'teal'
                    station_color = 'black'
                    shape = '^'

                GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3
                name = station[0:-4]
                station_names.append(name)

                residual = 100*(GPS_mean_vel-mean_model_vel)/GPS_mean_vel

                if residual > max_val:
                    max_val = residual
                if residual < min_val:
                    min_val = residual

                plt.scatter(loc, residual, c='black', s=300, zorder=10, marker=shape, edgecolors='black')
                loc+=1



        # plot
        plt.grid()
        plt.ylabel("Residual (% diff)", fontsize=40)
        plt.xlabel("Station", fontsize=40)
        plt.ylim([min_val-100, max_val+1000])
        plt.xticks(np.arange(loc), station_names, rotation ='horizontal', fontsize=30)
        plt.yticks(fontsize=30)
        plt.yscale('symlog')

        file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_mean_time_res"
        file_name += self.model_name.replace("/", "_") + ".png"
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
        plt.show() 


    # Plot the model data residuals points on the groundsurf mapview
    def plot_profile_res_mean_time_groundsurf(self, mean_times, shift_time=None, stations=None, station_locs=None, x_lim=None, y_lim=None,
                                              CR=False, CR_x=100, CR_y=None, CR_inner=None, CR_disp=None, inner_CR_disp=None, source=False, second_source=None, 
                                              source_disp=None, second_source_disp=None):

        # get base profile
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # get mean model velocities at each point
        mean_model_vel_list = []
        mean_station_vel_list = []
        station_names = []
        color_list = []
        included_station_locs = []
        for point, station in zip(station_locs, stations):
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)

            # get mean  modelvelocity
            model_vel_list = []
            for t in mean_times:
                model_time = self.get_timesteps(t)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
                point_idx = (np.abs(x_0 - r)).argmin() #get point index
                model_vel_list.append(vel_z[point_idx]*self.ms_to_mmyr)
            mean_model_vel = np.mean(np.asarray(model_vel_list), axis=0)

            # get mean station velocity
            df = pd.read_csv(station)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values
            
            time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)

            # if the station has data in the time cut, append to all of the lists
            if len(GPS_vel[time_cut]) > 0:

                if station in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                    station_color = 'indigo'
                else:
                    station_color = 'teal'

                color_list.append(station_color)
                GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3
                mean_station_vel_list.append(GPS_mean_vel)
                mean_model_vel_list.append(mean_model_vel)
                name = station[0:-4]
                station_names.append(name)
                included_station_locs.append(point)

            else:
                print("No data for station", station[0:-4], "in time cut")



        # get residuals
        # residuals = np.asarray(mean_station_vel_list)-np.asarray(mean_model_vel_list)
        residuals = 100*(np.asarray(mean_station_vel_list)-np.asarray(mean_model_vel_list))/np.asarray(mean_station_vel_list)
                

        # make groundsurf plot
        plt.xlabel("X [km]", fontsize=40)
        plt.ylabel("Y [km]", fontsize=40)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=40)
         # set plot limits
        if x_lim != None:
            plt.xlim(x_lim)
        if y_lim != None:
            plt.ylim(y_lim)

        plt.gca().set_aspect('equal')

        from matplotlib import colors
        color_map='seismic'
        # divnorm=colors.TwoSlopeNorm(vmin=-50., vcenter=0., vmax=50)
        divnorm=matplotlib.colors.SymLogNorm(linthresh=5, linscale=1.0, vmin=-1e3, vmax=1e3)

        for point, res in zip(included_station_locs, residuals):
            plt.scatter(point[0], point[1], c=res, marker='s', s=200, edgecolor='black', linewidths=2, cmap=color_map, norm=divnorm, zorder=5)

        # plot CR
        if CR:
            CR_x_points, CR_y_points = get_CR_points(CR_x, CR_y)
            plt.plot(CR_x_points+CR_disp[0], CR_y_points+CR_disp[1], c='black', linestyle='dashed', linewidth=5)
            
        # plot inner CR
        if CR_inner != None:
            CR_x_inner, CR_y_inner = get_CR_points(CR_inner[0], CR_inner[1])
            plt.plot(CR_x_inner+inner_CR_disp[0], CR_y_inner+inner_CR_disp[1], c='black', linestyle='dashdot', linewidth=3)
        
        # plot source
        if source:
            source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
            if source_disp == None:
                plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot', linewidth=4.5, zorder=3)
            else:
                plt.plot(source_x_points+source_disp[0], source_y_points+source_disp[1], c='black', linestyle='dashdot', linewidth=4.5, zorder=2)
            if second_source != None:
                second_source_x_points, second_source_y_points = get_CR_points(second_source[0], second_source[1])
                if second_source_disp == None:
                    plt.plot(second_source_x_points, second_source_y_points, c='black', linestyle='dashdot')
                else:
                    plt.plot(second_source_x_points+second_source_disp[0], second_source_y_points+second_source_disp[1], c='black', linestyle='dashdot')

        cbar = plt.colorbar()
        # cbar.set_label(label="Residual (mm/yr)", size=40)
        cbar.set_label(label="Residual (% diff)", size=40)
        cbar.ax.tick_params(labelsize=30)

        plt.show()



        

    # get the average z velocity of a set of points within a specified elliptical or elliptical shell region. Time step and at least one
    # set of minor/major axis dimensions [km] must be passed in, along with an optional second major/minor axes to define an inner ellipse
    # to cut a shell.
    def get_average_vel_area(self, time_step, outer_axes, offset=[0,0], inner_axes=None):

        if time_step not in self.source_velocity:

            # start by reading h5 file
            mesh_width = 150e3 #width of mesh from 0 to edge

            #first need to get model time
            model_time = self.get_timesteps(time_step)
            # print(time_step, model_time)

            #get the file path
            if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
                path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
            elif self.model.tr == 0.1:
                path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
            else:
                path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 

            #prepare lists for getting data from hdf5 files
            with h5py.File(path, "r") as f:
                group_geometry = f['geometry']
                group_vert_fields = f['vertex_fields']
                
                points = group_geometry['vertices'] #shape: point_num, xyz
                velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
            
                x = points[:][:,0]
                y = points[:][:,1]

                vel_z = velocities[model_time][:][:,2]

                # get indices of points in region
                idx_list = []
                for i in range(len(x)):
                    if inner_axes == None:
                        if (x[i]-offset[0]*1e3)**2/(outer_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(outer_axes[1]*1e3)**2 <= 1:
                            idx_list.append(i)
                    else:
                        if(x[i]-offset[0]*1e3)**2/(outer_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(outer_axes[1]*1e3)**2 <= 1 and (x[i]-offset[0]*1e3)**2/(inner_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(inner_axes[1]*1e3)**2 > 1:
                            idx_list.append(i)

                avg_vel = np.mean(vel_z[np.asarray(idx_list)]*self.ms_to_mmyr)

                self.source_velocity[time_step] = avg_vel # add the average velocity to the dictionary

                return avg_vel
        else:
            return self.source_velocity[time_step]

    # get the average z velocity of points some distance off of (for now) a line going through the x axis of the model 
    def get_average_vel_line(self, time_step, bounds, offset=0, dy=1.0):

        # start by reading h5 file
        mesh_width = 150e3 #width of mesh from 0 to edge

        #first need to get model time
        model_time = self.get_timesteps(time_step)
        # print(time_step, model_time)

        #get the file path
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 

        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]

            vel_z = velocities[model_time][:][:,2]

            # get indices of points in the region
            idx_list = []
            for i in range(len(x)):
                if x[i] >= bounds[0]*1e3 and x[i] <= bounds[1]*1e3 and (y[i] <= dy*1e3+offset*1e3 and y[i] >= offset*1e3-dy*1e3):
                    idx_list.append(i)
            print(len(idx_list))

            avg_vel = np.mean(vel_z[np.asarray(idx_list)]*self.ms_to_mmyr)
            return avg_vel
    
    # function to plot velocity averages on line over time
    def plot_vel_line_avg_time(self, time_steps, CR_x, norm=False, pressure_func_file=None, save=False):

        # get average velocities for center and shoulder
        center_vel = []
        shoulder_vel = []
        for t in time_steps:
            center_vel.append(self.get_average_vel_line(t, [-self.r_x, self.r_x], offset=self.y_off, dy=5.0))
            shoulder_vel.append((self.get_average_vel_line(t, [-CR_x, -self.r_x], dy=5.0)+self.get_average_vel_line(t, [self.r_x, CR_x], dy=5.0))/2)

        norm_center = max(center_vel)
        norm_shoulder = max(shoulder_vel)

        # set up plot
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time [years]", fontsize=self.label_fontsize)
        if norm:
            ax1.set_ylabel("Normalized Velocity", fontsize=self.label_fontsize)
        else:
            ax1.set_ylabel("Vertical Velocity [mm/yr]", fontsize=self.label_fontsize)
        ax1.grid()
        ax1.tick_params(axis='y', labelsize=self.tick_fontsize)
        ax1.tick_params(axis='x', labelsize=self.tick_fontsize)

        # plot

        if norm:
                ax1.plot(time_steps, np.asarray(center_vel)/max(norm_center), lw=self.lw, label="Center Line Area")
                ax1.plot(time_steps, np.asarray(shoulder_vel)/max(norm_shoulder), lw=self.lw, label="Shoulder Line Area")
        else:    
            ax1.plot(time_steps, np.asarray(center_vel), lw=self.lw, label="Center Line Area")
            ax1.plot(time_steps, np.asarray(shoulder_vel), lw=self.lw, label="Shoulder Line Area")


         # plot pressure function
        if pressure_func_file != None:
            df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            pressure_times = df['times']
            pressure_values = df['pressures']

            ax2 = ax1.twinx()
            ax2.set_ylabel("Pressure Function Amplitude Change", fontsize=self.label_fontsize)
            ax2.set_xlim([min(np.asarray(time_steps)), max(np.asarray(time_steps))])

            ax2.plot(pressure_times, np.asarray(pressure_values)-1, linewidth=self.lw, color='black', linestyle='dashed')
            ax2.tick_params(axis='y', labelsize=self.axes_fontsize)
        
        
        ax1.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=self.label_fontsize)

        ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
        ax1.patch.set_visible(False) # hide the 'canvas'
        
        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/mean_points_line_time_"
            file_name += self.model_name.replace("/", "_") + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

        
    # function to plot velocity averages over time
    def plot_vel_avg_time(self, time_steps, center, shoulder_outer, shoulder_inner, norm=False, pressure_func_file=None, save=False):

        # get average velocities for center and shoulder
        center_vel = []
        shoulder_vel = []
        for t in time_steps:
            center_vel.append(self.get_average_vel_area(t, center))
            shoulder_vel.append(self.get_average_vel_area(t, shoulder_outer, inner_axes=shoulder_inner))
        
        norm_center = max(center_vel)
        norm_shoulder = max(shoulder_vel)

        # set up plot
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time [years]", fontsize=self.label_fontsize)
        if norm:
            ax1.set_ylabel("Normalized Velocity", fontsize=self.label_fontsize)
        else:
            ax1.set_ylabel("Vertical Velocity [mm/yr]", fontsize=self.label_fontsize)
        ax1.grid()
        ax1.tick_params(axis='y', labelsize=self.tick_fontsize)
        ax1.tick_params(axis='x', labelsize=self.tick_fontsize)

        # plot

        if norm:
                ax1.plot(time_steps, np.asarray(center_vel)/max(norm_center), lw=self.lw, label="Center Area (a="+str(center[0])+"km,b="+str(center[1])+"km)")
                ax1.plot(time_steps, np.asarray(shoulder_vel)/max(norm_shoulder), lw=self.lw, label="Shoulder Area (a="+str(shoulder_inner[0])+"-"+str(shoulder_outer[0])+"km,b="+str(shoulder_inner[1])+"-"+str(shoulder_outer[1])+"km)")
        else:    
            ax1.plot(time_steps, np.asarray(center_vel), lw=self.lw, label="Center Area (a="+str(center[0])+"km,b="+str(center[1])+"km)")
            ax1.plot(time_steps, np.asarray(shoulder_vel), lw=self.lw, label="Shoulder Area (a="+str(shoulder_inner[0])+"-"+str(shoulder_outer[0])+"km,b="+str(shoulder_inner[1])+"-"+str(shoulder_outer[1])+"km)")


         # plot pressure function
        if pressure_func_file != None:
            df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            pressure_times = df['times']
            pressure_values = df['pressures']

            ax2 = ax1.twinx()
            ax2.set_ylabel("Pressure Function Amplitude Change", fontsize=self.label_fontsize)
            ax2.set_xlim([min(np.asarray(time_steps)), max(np.asarray(time_steps))])

            ax2.plot(pressure_times, np.asarray(pressure_values)-1, linewidth=self.lw, color='black', linestyle='dashed')
            ax2.tick_params(axis='y', labelsize=self.axes_fontsize)
        
        
        ax1.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=self.label_fontsize)

        ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
        ax1.patch.set_visible(False) # hide the 'canvas'
        
        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/mean_points_time_"
            file_name += self.model_name.replace("/", "_") + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

    
    # plot a parameter (disp, vel) over the full 2D ground surface and then plot profiles from that parameter
    # if profile is set to true, a profile will be extracted and plotted based off the theta and offset parameters passed in
    def read_plot_groundsurf(self, time_step, mean_time_steps=None, parameter='vz', som_width=False, CR=False, CR_x=100, CR_y=None, CR_inner=None, envelope=None, source=False, 
                             second_source=None, source_disp=None, second_source_disp=None, CR_disp=None, inner_CR_disp=None, profile=False, x_lim=None, 
                             y_lim=None, theta=None, save=False, in_points_list=None, out_points_list=None, station_files=None,
                             plot_caldera=False, log_scale=True, vel_lim=None, contours=True):

        #first need to get model time
        if mean_time_steps == None:
            model_time = self.get_timesteps(time_step)
        else:
            model_time_list = []
            for t in mean_time_steps:
                model_time_list.append(self.get_timesteps(t))
            model_time = np.asarray(model_time_list)
        # print(time_step, model_time)

        #get the file path
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 

        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            if mean_time_steps == None:
                disp_x = displacements[model_time][:][:,0]
                disp_y = displacements[model_time][:][:,1]
                disp_z = displacements[model_time][:][:,2]
                
                vel_x = velocities[model_time][:][:,0]
                vel_y = velocities[model_time][:][:,1]
                vel_z = velocities[model_time][:][:,2]
            else:
                # get arrays of 2D data at each time step
                disp_x_list = []
                disp_y_list = []
                disp_z_list = []
                vel_x_list = []
                vel_y_list = []
                vel_z_list = []

                for mt in model_time:
                    disp_x_list.append(displacements[mt][:][:,0])
                    disp_y_list.append(displacements[mt][:][:,1])
                    disp_z_list.append(displacements[mt][:][:,2])
                    
                    vel_x_list.append(velocities[mt][:][:,0])
                    vel_y_list.append(velocities[mt][:][:,1])
                    vel_z_list.append(velocities[mt][:][:,2])


            #get radial components
            # angle = np.arctan(y/x)
            # disp_r = disp_x*np.cos(angle)+disp_y*np.sin(angle)
            # disp_theta = -disp_x*np.sin(angle)+disp_y*np.cos(angle)
        
            # vel_r = vel_x*np.cos(angle)+vel_y*np.sin(angle)
            # vel_theta = -vel_x*np.sin(angle)+vel_y*np.cos(angle)

            #get profile if needed
            if profile:
                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                  vel_x_prof, vel_y_prof, vel_z_prof, 
                  vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)

            #Select data being plotted
            if parameter == 'dx':
                if mean_time_steps == None:
                    plot_data = disp_x
                else:
                    plot_data = np.asarray(disp_x_list)
                cbar_label = "Displacement X [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_x_prof
            elif parameter == 'dy':
                if mean_time_steps == None:
                    plot_data = disp_y
                else: 
                    plot_data = np.asarray(disp_y_list)
                cbar_label = "Displacement Y [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_y_prof
            elif parameter == 'dz':
                if mean_time_steps == None:
                    plot_data = disp_z
                else:
                    plot_data = np.asarray(disp_z_list)
                cbar_label = "Displacement Z [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_z_prof
            elif parameter == 'vx':
                if mean_time_steps == None:
                    plot_data = vel_x
                else:
                    plot_data = np.asarray(vel_x_list)
                cbar_label = "Velocity X [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_x_prof
            elif parameter == 'vy':
                if mean_time_steps == None:
                    plot_data = vel_y
                else:
                    plot_data = np.asarray(vel_y_list)
                cbar_label = "Velocity Y [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_y_prof
            elif parameter == 'vz':
                if mean_time_steps == None:
                    plot_data = vel_z
                else:
                    plot_data = np.asarray(vel_z_list)
                cbar_label = "Velocity Z [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_z_prof
            else:
                print("Parameter \""+parameter+"\" is not supported. Will not continue.")
                return
            
            if mean_time_steps != None:
                # take the mean of the plot data
                plot_data_mean = np.mean(plot_data, axis=0)

            #interpolate
            grid_x, grid_y = np.mgrid[-self.mesh_width:self.mesh_width:1000j, -self.mesh_width:self.mesh_width:1000j]
            
            if mean_time_steps == None:
                z = interpolate.griddata((x/1e3,y/1e3), plot_data, (grid_x/1e3, -grid_y/1e3), method='cubic')
            else:
                z = interpolate.griddata((x/1e3,y/1e3), plot_data_mean, (grid_x/1e3, -grid_y/1e3), method='cubic')



            #plot
            plt.figure()
            #color_map = plt.cm.get_cmap('RdBu').reversed()
            color_map='seismic'
            if log_scale:
                plt.imshow(z.T*conversion, extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3),
                            cmap=color_map, norm=matplotlib.colors.SymLogNorm(linthresh=0.5, linscale=1.0, vmin=-80, vmax=80))
            else:
                from matplotlib import colors
                if vel_lim == None:
                    divnorm=colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=60)
                else: 
                    divnorm=colors.TwoSlopeNorm(vmin=vel_lim[0], vcenter=0., vmax=vel_lim[1])

                im = plt.imshow(z.T*conversion, extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3),
                            cmap=color_map, norm=divnorm)
                # color_map = plt.cm.get_cmap('RdBu').reversed()
                # levels = np.arange(-10, 35, 5)
                if contours == True:
                    ctr = plt.contour(z.T*conversion, [-10, -5, 0, 5, 10, 20, 30, 40, 50], origin='upper', colors=['black', 'black', 'black', 'black', 'black'],
                                    extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3))

            plt.xlabel("X [km]", fontsize=40)
            plt.ylabel("Y [km]", fontsize=40)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=40)
            if log_scale:
                cbar = plt.colorbar()
            else:
                cbar = plt.colorbar(im)
                # cbar = plt.colorbar(im, orientation='horizontal')
                if contours:
                    cbar.add_lines(ctr)
            cbar.set_label(label="Velocity (mm/yr)", size=40)
            cbar.ax.tick_params(labelsize=30)
            # if in_points_list != None:
            #     plt.plot(np.zeros(121), np.arange(-60, 61, 1)+self.y_off/1e3, linewidth=3.5, c="black", zorder=1)
            inner_color = 'indigo'
            outer_color = 'teal'
            if station_files != None:
                shift_time=1986
                points_list = in_points_list+out_points_list
                for station, point in zip(station_files, points_list):
                    # get mean station velocity
                    df = pd.read_csv(station)
                    GPS_times = df['time[yrs]'].values
                    GPS_vel = df['velocity[m/yr]'].values
                    time_cut = (GPS_times >= mean_time_steps[0]-500+shift_time) & (GPS_times <= mean_time_steps[1]-500+shift_time) 

                    if len(GPS_vel[time_cut]) > 0:
                        mean_vel =  np.mean(GPS_vel[time_cut])*1e3
                        if station in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                            plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=mean_vel, marker='s', s=600, edgecolor='black', linewidths=4, zorder=10, 
                                        cmap=color_map, norm=divnorm)
                        else:
                            plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=mean_vel, marker='^', s=600, edgecolor='black', linewidths=4, zorder=10, 
                                        cmap=color_map, norm=divnorm)

            else:
                if in_points_list != None:
                    for point in in_points_list:
                        plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=inner_color, marker='s', s=600, edgecolor='black', linewidths=4, zorder=2)
                if out_points_list != None:
                    for point in out_points_list:
                        plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=outer_color, marker='s', s=600, edgecolor='black', linewidths=4)
                
            if plot_caldera:
                df_caldera = pd.read_csv("shifted_caldera.csv", sep=',', header=None)
                plt.plot(df_caldera[0]+self.x_off/1e3, df_caldera[1]+self.y_off/1e3, linewidth=3)

            #get points of contour enclosing the sombrero width
            if som_width:
                if parameter=='vz':
                    x_min_list, y_min_list = self.get_som_width_contour(time_step)
                    plt.scatter(np.asarray(x_min_list)/1e3, np.asarray(y_min_list)/1e3, c='black', s=10)
                else:
                    print("Sombrero Width profiles are currently only implemented for vz.")
            
            if CR_disp == None:
                CR_disp = (0, 0)

            # plot CR
            if CR:
                CR_x_points, CR_y_points = get_CR_points(CR_x, CR_y)
                plt.plot(CR_x_points+CR_disp[0], CR_y_points+CR_disp[1], c='black', linestyle='dashed', linewidth=5)
            
            # plot inner CR
            if CR_inner != None:
                CR_x_inner, CR_y_inner = get_CR_points(CR_inner[0], CR_inner[1])
                plt.plot(CR_x_inner+inner_CR_disp[0], CR_y_inner+inner_CR_disp[1], c='black', linestyle='dashdot', linewidth=3)

            # plot the envelope
            if envelope != None:
                envelope_x, envelope_y = get_CR_points(envelope[0], envelope[1])
                plt.plot(envelope_x, envelope_y, linestyle='dashdot', c='black')

            # plot source
            if source:
                source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
                if source_disp == None:
                    plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot', linewidth=4.5, zorder=3)
                else:
                    plt.plot(source_x_points+source_disp[0], source_y_points+source_disp[1], c='black', linestyle='dashdot', linewidth=4.5, zorder=2)
                if second_source != None:
                    second_source_x_points, second_source_y_points = get_CR_points(second_source[0], second_source[1])
                    if second_source_disp == None:
                        plt.plot(second_source_x_points, second_source_y_points, c='black', linestyle='dashdot', linewidth=4.5)
                    else:
                        plt.plot(second_source_x_points+second_source_disp[0], second_source_y_points+second_source_disp[1], c='black', linestyle='dashdot', linewidth=4.5)
            
            
            # set plot limits
            if x_lim != None:
                plt.xlim(x_lim)
            if y_lim != None:
                plt.ylim(y_lim)

            #plot profile line if needed
            if profile:
                x_i = np.linspace(-self.mesh_width, self.mesh_width, len(x_prof))
                if theta == None:
                    y_i = np.zeros(len(x_i))
                else:
                    x_i_temp = x_i
                    x_i = np.cos(theta)*x_i_temp 
                    y_i = np.sin(theta)*x_i_temp 
                    
                x_i += self.x_off
                y_i += self.y_off
                plt.plot(x_i/1e3, y_i/1e3, c="black")

            if save:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/plot_groundsurface_gs"
                file_name += self.model_name.replace("/", "_") + parameter + "_"+str(time_step)+".png"
                plt.savefig(file_name, bbox_inches="tight")

            #plot profile
            if profile:
                plt.figure()
                plt.plot(x_i_temp/1e3, prof_data*conversion, linewidth=self.lw)
                plt.xlabel("Distance from center of profile [km]", fontsize=self.axes_fontsize)
                plt.ylabel(cbar_label, fontsize=self.axes_fontsize)
                plt.xticks(fontsize=self.tick_fontsize)
                plt.yticks(fontsize=self.tick_fontsize)
                plt.grid()

                if save:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/plot_groundsurface_prof"
                    file_name += self.model_name.replace("/", "_") + parameter + ".png"
                    plt.savefig(file_name, bbox_inches="tight")

            plt.show()

    # Plot 2D groundsurface data like the above function averaged over given time window
    # Default dt is 1 year. 
    def plot_groundsurf_timeaverage(self, time_window, dt=1, parameter='vz', CR=False, CR_x=100, CR_y=None, CR_inner=None, envelope=None, source=False, 
                             second_source=None, source_disp=None, second_source_disp=None, x_lim=None, y_lim=None,save=False):
        
        # Read h5 file for each time step
        mesh_width = 150e3 #width of mesh from 0 to edge

        #first need to get model times
        times = np.arange(time_window[0], time_window[1], dt)
        model_times_list = []
        for t in times:
            model_times_list.append(self.get_timesteps(t))
        model_times = np.asarray(model_times_list)
        # print(times[0], model_times[0])

        #get the file path
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 


        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            # get arrays of 2D data at each time step
            disp_x_list = []
            disp_y_list = []
            disp_z_list = []
            vel_x_list = []
            vel_y_list = []
            vel_z_list = []

            for mt in model_times:
                disp_x_list.append(displacements[mt][:][:,0])
                disp_y_list.append(displacements[mt][:][:,1])
                disp_z_list.append(displacements[mt][:][:,2])
                
                vel_x_list.append(velocities[mt][:][:,0])
                vel_y_list.append(velocities[mt][:][:,1])
                vel_z_list.append(velocities[mt][:][:,2])

             #Select data being plotted
            if parameter == 'dx':
                plot_data = np.asarray(disp_x_list)
                cbar_label = "Displacement X [mm]"
                conversion = 1e3
            elif parameter == 'dy':
                plot_data = np.asarray(disp_y_list)
                cbar_label = "Displacement Y [mm]"
                conversion = 1e3
            elif parameter == 'dz':
                plot_data = np.asarray(disp_z_list)
                cbar_label = "Displacement Z [mm]"
                conversion = 1e3
            elif parameter == 'vx':
                plot_data = np.asarray(vel_x_list)
                cbar_label = "Velocity X [mm/yr]"
                conversion = self.ms_to_mmyr
            elif parameter == 'vy':
                plot_data = np.asarray(vel_y_list)
                cbar_label = "Velocity Y [mm/yr]"
                conversion = self.ms_to_mmyr
            elif parameter == 'vz':
                plot_data = np.asarray(vel_z_list)
                cbar_label = "Velocity Z [mm/yr]"
                conversion = self.ms_to_mmyr
            else:
                print("Parameter \""+parameter+"\" is not supported. Will not continue.")
                return
            
            # take the mean of the plot data
            plot_data_mean = np.mean(plot_data, axis=0)

            #interpolate
            grid_x, grid_y = np.mgrid[-mesh_width:mesh_width:1000j, -mesh_width:mesh_width:1000j]

            z = interpolate.griddata((x/1e3,y/1e3), plot_data_mean, (grid_x/1e3, grid_y/1e3), method='cubic')

            #plot
            plt.figure()
            color_map='seismic'
            plt.imshow(z.T*conversion, extent=(-mesh_width/1e3,mesh_width/1e3,-mesh_width/1e3,mesh_width/1e3),
                        cmap=color_map, norm=matplotlib.colors.SymLogNorm(linthresh=0.5, linscale=1.0, vmin=-80, vmax=80))
            plt.xlabel("X [km]", fontsize=self.label_fontsize)
            plt.ylabel("Y [km]", fontsize=self.label_fontsize)
            plt.xticks(fontsize=self.tick_fontsize)
            plt.yticks(fontsize=self.tick_fontsize)
            cbar = plt.colorbar()
            cbar.set_label(label=cbar_label, size=self.label_fontsize)
            cbar.ax.tick_params(labelsize=self.tick_fontsize)


            # plot CR
            if CR:
                CR_x_points, CR_y_points = get_CR_points(CR_x, CR_y)
                plt.plot(CR_x_points, CR_y_points, c='black', linestyle='dashed')
            
            # plot inner CR
            if CR_inner != None:
                CR_x_inner, CR_y_inner = get_CR_points(CR_inner[0], CR_inner[1])
                plt.plot(CR_x_inner, CR_y_inner, c='black')

            # plot the envelope
            if envelope != None:
                envelope_x, envelope_y = get_CR_points(envelope[0], envelope[1])
                plt.plot(envelope_x, envelope_y, linestyle='dashdot', c='black')

            # plot source
            if source:
                source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
                if source_disp == None:
                    plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot')
                else:
                    plt.plot(source_x_points+source_disp[0], source_y_points+source_disp[1], c='black', linestyle='dashdot')
                if second_source != None:
                    second_source_x_points, second_source_y_points = get_CR_points(second_source[0], second_source[1])
                    if second_source_disp == None:
                        plt.plot(second_source_x_points, second_source_y_points, c='black', linestyle='dashdot')
                    else:
                        plt.plot(second_source_x_points+second_source_disp[0], second_source_y_points+second_source_disp[1], c='black', linestyle='dashdot')
            
            # set plot limits
            if x_lim != None:
                plt.xlim(x_lim)
            if y_lim != None:
                plt.ylim(y_lim)


            if save:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/plot_groundsurface_timeaverage"
                file_name += self.model_name.replace("/", "_") + parameter + "_"+str(time_window[0])+"-"+str(time_window[1])+".png"
                plt.savefig(file_name, bbox_inches="tight")

            plt.show()

            


    
    # Read model velocity component and compare with chosen Yellowstone area GPS
    # station. List of GPS stations can be found in ~/Research/Yellowstone/Yellowstone_GPS.ods.
    # Model data will be taken at start_time and read for time duration of chosen Yellowstone
    # GPS station. If station duration is longer than the model time left after the start_time
    # the function will crash. Position should be a tuple in km specifying cartesian point
    # on model to collect velocity data. 
    def compare_point_GPS(self, start_time, position, station_name, component="UP", save=False):

        # load in GPS Station
        GPS_Path = "/home/grantblock/Research/Yellowstone/GPS_Data/"+station_name+".NA.tenv3.txt"
        GPS_station = ReadGPS(GPS_Path)
        
        # get time steps
        GPS_dur = max(GPS_station.dec_years)-min(GPS_station.dec_years)
        time_steps = np.arange(start_time, start_time+GPS_dur, 0.1)

        # loop through time steps and collect model vel data at specified point
        point_disp_list = []
        for t in time_steps:
            model_time = self.get_timesteps(t)

            #get the file path
            if self.model.tr == 0.1:
                path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
            else:
                path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 
        
            #prepare lists for getting data from hdf5 files
            with h5py.File(path, "r") as f:
                group_geometry = f['geometry']
                group_vert_fields = f['vertex_fields']
                
                points = group_geometry['vertices'] #shape: point_num, xyz
                displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
                velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
            
                x = points[:][:,0]
                y = points[:][:,1]
                
                disp_x = displacements[model_time][:][:,0]
                disp_y = displacements[model_time][:][:,1]
                disp_z = displacements[model_time][:][:,2]
                

                # convert coordinates to r and theta
                r = np.sqrt((position[0]*1e3)**2 + (position[1]*1e3)**2)
                if position[0] != 0:
                    theta = np.arctan(position[1]/position[0])
                else:
                    if position[1] >= 0 :
                        theta = np.pi/2.
                    else:
                        theta = 3*np.pi/2.

                # interpolate line along theta
                mesh_width=150e3
                x_i = np.linspace(-mesh_width, mesh_width, 11615)
                x_i_temp = x_i
                x_i = np.cos(theta)*x_i_temp 
                y_i = np.sin(theta)*x_i_temp 

                disp_x_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_x), np.asarray([x_i, y_i]).T, method='cubic')
                disp_y_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_y), np.asarray([x_i, y_i]).T, method='cubic')
                disp_z_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_z), np.asarray([x_i, y_i]).T, method='cubic')

                # get index of point
                point_idx = (np.abs(x_i-r)).argmin()
                if component=="NORTH":
                    point_disp_list.append(disp_x_interp[point_idx]) # North and east probably have to be rotated to fit with actual data orientation, TODO!
                elif component=="EAST":
                    point_disp_list.append(disp_y_interp[point_idx])
                else:
                    point_disp_list.append(disp_z_interp[point_idx])
        
        # plot model displacements at point and compare to GPS displacements
        plt.grid()
        plt.xlabel("Time [yrs]", fontsize=self.label_fontsize)
        if component=="NORTH":
            plt.ylabel("North Displacement [mm]", fontsize=self.label_fontsize)
        elif component=="EAST":
            plt.ylabel("East Displacement [mm]", fontsize=self.label_fontsize)
        else:
            plt.ylabel("Vertical Displacement [mm]", fontsize=self.label_fontsize)

        plt.xticks(fontsize=self.axes_fontsize)
        plt.yticks(fontsize=self.axes_fontsize)

        time_diff = GPS_station.dec_years[0]-time_steps[0]
        if component=="NORTH":
            plt.scatter(GPS_station.dec_years-time_diff, GPS_station.northing-np.mean(GPS_station.northing[1:10]), s=3, c="black", label="GPS data")
        elif component=="EAST":
            plt.scatter(GPS_station.dec_years-time_diff, GPS_station.easting-np.mean(GPS_station.easting[1:10]), s=3, c="black", label="GPS data")
        else:
            plt.scatter(GPS_station.dec_years-time_diff, GPS_station.up-np.mean(GPS_station.up[1:10]), s=3, c="black", label="GPS data")

        plt.plot(time_steps, np.asarray(point_disp_list)*1e3-point_disp_list[0]*1e3, linewidth=self.lw, c="blue", label="Model")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.label_fontsize)
        if save:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/compare_point_GPS"
                file_name += self.model_name.replace("/", "_") +"_"+GPS_station.name+ ".png"
                plt.savefig(file_name, bbox_inches="tight")

        plt.show()

    # Make plot with velocity of model point, observed GPS velocity and the pressure function 
    def plot_point_station(self, times, points_inner, points_outer, no_station_points_outer=[], GPS_vel_file_inner=[], 
                           GPS_vel_file_outer=[], GPS_vel_file_no_model_outer=[], inner_colors=[], outer_colors=[], outer_colors_no_model=[], 
                           outer_colors_no_station=[], inner_ls=[], outer_ls=[], outer_ls_no_model=[],
                           inner_markers=[], outer_markers=[], outer_markers_no_model=[], shift_time=1999.326, mult_factor=1,
                           save_name=None):

        # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)
        
        # set up plot
        # fig, ax = plt.subplots(2, 1, figsize=(14.8, 6),sharex=True)
        fig, ax = plt.subplots(2, 1, figsize=(17, 6),sharex=True)
        # ax[0].set_ylabel("Vertical Velocity (mm/yr)", fontsize=30)
        fig.text(0.04, 0.5, "Vertical Velocity (mm/yr)", fontsize=30, va='center', rotation='vertical')
        ax[0].grid()
        ax[0].tick_params(axis='y', labelsize=30)
        ax[1].grid()
        # ax[1].set_ylabel("Vertical Velocity (mm/yr)", fontsize=25)
        ax[1].set_xlabel("Time (yrs)", fontsize=30)
        ax[1].tick_params(axis='y', labelsize=30)
        ax[1].tick_params(axis='x', labelsize=30)

        # loop through inner and outer points and plot
        list_itr = 0
        for point in points_inner:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            
            ax[0].plot(times-500+shift_time, np.asarray(vel_array)*self.ms_to_mmyr*mult_factor, c=inner_colors[list_itr], lw=10, alpha=0.5, label=GPS_vel_file_inner[list_itr][0:4]+" model")
            list_itr+=1

        list_itr = 0
        for point in points_outer:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            
            ax[1].plot(times-500+shift_time, np.asarray(vel_array)*self.ms_to_mmyr*mult_factor, c=outer_colors[list_itr], lw=10, alpha=0.5, label=GPS_vel_file_outer[list_itr][0:4]+" model")
            list_itr+=1

        list_itr = 0
        for point in no_station_points_outer:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            
            ax[1].plot(times-500+shift_time, np.asarray(vel_array)*self.ms_to_mmyr*mult_factor, c=outer_colors_no_station[list_itr], lw=10, alpha=0.5, label="Smallest Ratio Location")
            list_itr+=1

        # iterate through the inner and outer GPS velocity data and plot
        list_itr = 0
        for file in GPS_vel_file_inner:
            df = pd.read_csv(file)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values

            ax[0].plot(GPS_times, GPS_vel*1e3, c=inner_colors[list_itr], linestyle=inner_ls[list_itr], marker=inner_markers[list_itr], 
            markevery=5, markersize=10, markeredgecolor='black', markeredgewidth=0.6, lw=4, label=GPS_vel_file_inner[list_itr][0:4])
            list_itr+=1

        list_itr = 0
        for file in GPS_vel_file_outer:
            df = pd.read_csv(file)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values

            ax[1].plot(GPS_times, GPS_vel*1e3, c=outer_colors[list_itr], linestyle=outer_ls[list_itr], marker=outer_markers[list_itr], 
                       markevery=5, markersize=10, markeredgecolor='black', markeredgewidth=0.6, lw=4, label=GPS_vel_file_outer[list_itr][0:4])
            list_itr+=1

        list_itr = 0
        for file in GPS_vel_file_no_model_outer:
            df = pd.read_csv(file)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values

            ax[1].plot(GPS_times, GPS_vel*1e3, c=outer_colors_no_model[list_itr], linestyle=outer_ls_no_model[list_itr], marker=outer_markers_no_model[list_itr], 
                       markevery=5, markersize=20, markeredgecolor='black', markeredgewidth=0.6, lw=10, label=GPS_vel_file_no_model_outer[list_itr][0:4])
            list_itr+=1

        # ax[0].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1, 1.1))
        # ax[1].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1, 1.1))
        
        plt.show()
        fig.savefig("../../Figures/Model_Obs.png", dpi=200, bbox_inches='tight')
        plt.close()

    
    # Make plot with velocity of model point, observed GPS velocity and the pressure function 
    def plot_point_station_avg(self, times, points_inner, points_outer, GPS_in_mean_file, 
                                GPS_out_mean_file, RG_file=None, shift_time=1999.326, mult_factor=1, plot_unscaled=False,
                                pressure_func_file=None, calc_metrics=False, write_files=False, save_name=None):
        

         # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)
        
        # set up plot
        # fig, ax = plt.subplots(2, 1, figsize=(14.8, 6),sharex=True)
        fig, ax = plt.subplots(2, 1, figsize=(17, 6),sharex=True)
        # ax[0].set_ylabel("Vertical Velocity (mm/yr)", fontsize=30)
        fig.text(0.04, 0.5, "Vertical Velocity (mm/yr)", fontsize=30, va='center', rotation='vertical')
        ax[0].grid()
        ax[0].tick_params(axis='y', labelsize=30)
        ax[1].grid()
        # ax[1].set_ylabel("Vertical Velocity (mm/yr)", fontsize=25)
        ax[1].set_xlabel("Time (yrs)", fontsize=30)
        ax[1].tick_params(axis='y', labelsize=30)
        ax[1].tick_params(axis='x', labelsize=30)


        # loop through inner and outer points and collect
        points_inner_list = []
        for point in points_inner:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            
            if type(mult_factor) != tuple and type(mult_factor) != list:
                points_inner_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor)
            else:
                points_inner_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor[0])

        points_outer_list = []   
        for point in points_outer:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0:
                r *= -1
            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
            
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            if type(mult_factor) != tuple and type(mult_factor) != list:
                points_outer_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor)
            else:
                points_outer_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor[1])


        # read inner and outer GPS mean files
        df_in = pd.read_csv(GPS_in_mean_file)
        GPS_in_times = df_in['time[yrs]'].values
        GPS_in_vel = df_in['velocity[m/yr]'].values

        df_out = pd.read_csv(GPS_out_mean_file)
        GPS_out_times = df_out['time[yrs]'].values
        GPS_out_vel = df_out['velocity[m/yr]'].values

        if RG_file != None:
            df_RG = pd.read_csv(RG_file)
            GPS_RG_times = (df_RG['time[yrs]'].values)[1:]
            GPS_RG_vel = (df_RG['velocity[m/yr]'].values)[1:]


            GPS_out_RG_vel = GPS_out_vel[GPS_out_times >= min(GPS_RG_times)] - GPS_RG_vel

        # get means of model point velocities
        points_inner_mean = np.mean(np.asarray(points_inner_list), axis=0)
        points_outer_mean = np.mean(np.asarray(points_outer_list), axis=0)

        # plot
        inner_color = 'indigo'#'royalblue'
        outer_color = 'teal'#'firebrick'

        if type(mult_factor) != tuple and type(mult_factor) != list:
            if mult_factor == 1: 
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$")
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{anti}$")
            else:
                l1 = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$, scaled x"+str(mult_factor))
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{anti}$, scaled x"+str(mult_factor))
        else:
            if mult_factor[0] == 1:
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$")
            else:
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, linestyle='dotted', label=r"Model $\langle V\rangle^{inner}$, scaled x"+str(mult_factor[0]))
                if plot_unscaled:
                   l1, =  ax[0].plot(times-500+shift_time, points_inner_mean/mult_factor[0], color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$, unscaled")


            if mult_factor[1] == 1:
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{anti}$")
            else:
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, linestyle='dotted', label=r"Model $\langle V\rangle^{anti}$, scaled x"+str(mult_factor[1]))
                if plot_unscaled:
                    ax[1].plot(times-500+shift_time, points_outer_mean/mult_factor[1], color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{anti}$, unscaled")




        l2, = ax[0].plot(GPS_in_times, GPS_in_vel*1e3, color=inner_color, lw=6, linestyle="--", label=r"Smoothed GPS $\langle V\rangle^{inner}$")
        
        ax[1].plot(GPS_out_times, GPS_out_vel*1e3, color=outer_color, lw=6, linestyle="--", label=r"GPS $\langle V\rangle^{anti}$")

        if RG_file != None:
            ax[1].plot(GPS_RG_times, GPS_out_RG_vel*1e3, color='gray', lw=6, linestyle="--", label=r"GPS $\langle V\rangle^{anti} - \langle V\rangle^{RG}$")
        
        # plot pressure function
        if pressure_func_file != None:
            df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            pressure_times = df['times']
            pressure_values = df['pressures']

            ax3 = ax[0].twinx()
            ax3.set_ylabel(r"$P(t)-P_0$ (kPa)", fontsize=25)
            ax3.set_xlim([min(np.asarray(times-500+shift_time)), max(np.asarray(times-500+shift_time))])

            l3, = ax3.plot(pressure_times-500+shift_time-1, (np.asarray(pressure_values)-1)*1e2, linewidth=self.lw, label="Applied Pressure Function", color='black', linestyle='dashed')
            ax3.tick_params(axis='y', labelsize=30)
        
            ax[0].set_zorder(ax3.get_zorder()+1) # put ax in front of ax2
            ax[0].patch.set_visible(False) # hide the 'canvas'
            ax[0].legend((l1, l2, l3), (l1.get_label(), l2.get_label(), l3.get_label()), handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.3, 1.1))
        else:
            ax[0].legend((l1, l2), (l1.get_label(), l2.get_label()), handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.15, 1.1))
        
        ax[1].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.15, 1.1))

        # calculate anti-corelation and ratio
        if calc_metrics:
            from scipy.stats import pearsonr
            from scipy.interpolate import interp1d

            unscaled_central = points_inner_mean/mult_factor[0]
            unscaled_outer = points_outer_mean/mult_factor[1]
            # cross correlation
            time_window = (times > 2004-shift_time+500) & (times <= 2016-shift_time+500)
            result_model = pearsonr(unscaled_central[time_window], unscaled_outer[time_window])

            # calculate ratio
            time_window = (times > 2005-shift_time+500) & (times <= 2007-shift_time+500)
            max_center_idx = np.argmax(unscaled_central[time_window])
            max_center_uplift = unscaled_central[time_window][max_center_idx]
            point_vel = unscaled_outer[time_window][max_center_idx]
            ratio = np.abs(max_center_uplift/point_vel)
            if point_vel > 0:
                print("Warning: outer velocity > 0, ratio invalid")

            # calculate data metrics
            time_window_GPS_in = (GPS_in_times > 2004) & (GPS_in_times <= 2016)
            time_window_GPS_out = (GPS_out_times >= min(GPS_in_times[time_window_GPS_in])) & (GPS_out_times <= max(GPS_in_times[time_window_GPS_in]))
            # interpolate the GPS_in time series
            interp_f = interp1d(GPS_in_times[time_window_GPS_in], GPS_in_vel[time_window_GPS_in])
            GPS_in_vel_interp = interp_f(GPS_out_times[time_window_GPS_out])
            result_data = pearsonr(GPS_in_vel_interp, GPS_out_vel[time_window_GPS_out])

            time_window_GPS_in = (GPS_in_times > 2005) & (GPS_in_times <= 2007)
            time_window_GPS_out = (GPS_out_times > 2005) & (GPS_out_times <= 2007)
            max_center_idx = np.argmax(GPS_in_vel[time_window_GPS_in])
            # print(GPS_in_times[time_window_GPS_in][max_center_idx], GPS_out_vel[np.argmin(np.abs(GPS_out_times -GPS_in_times[time_window_GPS_in][max_center_idx]))])
            max_center_uplift = GPS_in_vel[time_window_GPS_in][max_center_idx]
            point_vel =  GPS_out_vel[np.argmin(np.abs(GPS_out_times -GPS_in_times[time_window_GPS_in][max_center_idx]))]
            ratio_GPS = np.abs(max_center_uplift/point_vel)

            # Calculate data metrics with outer-RG
            time_window_GPS_in = (GPS_in_times > 2004) & (GPS_in_times <= 2016)
            time_window_GPS_RG = (GPS_RG_times >= min(GPS_in_times[time_window_GPS_in])) & (GPS_RG_times <= max(GPS_in_times[time_window_GPS_in]))
            interp_f = interp1d(GPS_in_times[time_window_GPS_in], GPS_in_vel[time_window_GPS_in])
            # print(GPS_in_times[time_window_GPS_in])
            # print(GPS_RG_times[time_window_GPS_RG])
            GPS_in_vel_interp = interp_f(GPS_RG_times[time_window_GPS_RG])
            result_data_RG = pearsonr(GPS_in_vel_interp, GPS_out_RG_vel[time_window_GPS_RG])

            # calculate CC between the smoothed data in (pressure function) and model in
            full_tw_model = (times >= min(GPS_in_times)-shift_time+500) & (times <= max(GPS_in_times)-shift_time+500)

            # result_full_model_pt = pearsonr(unscaled_central[full_tw_model], GPS_in_vel[1:]*1e3)

            # calculate misfits
            CC_misfit = (result_data[0] - result_model[0])/result_data[0]
            ratio_misfit = (ratio_GPS - ratio)/ratio_GPS
            total_misfit = 0.5*np.sqrt(CC_misfit**2 + ratio_misfit**2)

        
            print("Cross Correlation:", str(result_model[0]), "P-Value", str(result_model[1]), "Ratio:", str(ratio))
            print("GPS Cross Correlation:", str(result_data[0]), "Ratio:", str(ratio_GPS))
            print("CC misfit:", str(CC_misfit), "Ratio misfit:", str(ratio_misfit), "Total misfit:", str(total_misfit))
            # print("GPS Cross Correlation subtracting RG from outer:", str(result_data_RG[0]))
            # print("Full model, P(t) CC:", str(result_full_model_pt[0]))

        # write model output to csv files
        if write_files:
            unscaled_inner = points_inner_mean/mult_factor[0]
            unscaled_outer = points_outer_mean/mult_factor[1]

            parent_folder = './Data_fig2d/'
            file_name_inner = parent_folder+'model_inner.csv'
            file_name_outer = parent_folder+'model_outer.csv'

            shifted_time = times-500+shift_time

            # write inner
            f = open(file_name_inner, "w")
            f.write("time[yrs],velocity[mm/yr]\n")
            for i in range(len(shifted_time)):
                f.write(str(round(shifted_time[i], 3)) + "," + str(round(unscaled_inner[i], 5)) + "\n")
            f.close()

            #write outer
            f = open(file_name_outer, "w")
            f.write("time[yrs],velocity[mm/yr]\n")
            for i in range(len(shifted_time)):
                f.write(str(round(shifted_time[i], 3)) + "," + str(round(unscaled_outer[i], 5)) + "\n")
            f.close()

        
        plt.show()
        fig.savefig("../../Figures/Model_Obs_mean.png", dpi=200, bbox_inches='tight')
        plt.close()




        
       

    
    # Function to plot arbitrary points over time. 
    # Takes a list of points (in km), time array, and optional pressure
    # function file from which it can plot the applied pressure function from
    def plot_points_over_time(self, times, points, save=False, vel_scale=False, norm=False, pressure_func_file=None):

        # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)
        
        if vel_scale:
            d_uc=10e3
            scaling_param = self.model.P0/(self.model.Delta_P*d_uc)*1e-3 # in yr/mm
        else:
            scaling_param = 1

        # set up plot
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time [years]", fontsize=30)
        if norm:
            ax1.set_ylabel("Normalized Velocity", fontsize=30)
        else:
            if vel_scale:
                ax1.set_ylabel(r"$c_{vp}|V_z|$", fontsize=30)
            else:
                ax1.set_ylabel("Vertical Velocity [mm/yr]", fontsize=30)

        ax1.grid()
        ax1.tick_params(axis='y', labelsize=25)
        ax1.tick_params(axis='x', labelsize=25)

        # iterate over points
        norm_val = -1
        for point in points:
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)

            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)
                # print(t, model_time)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                        vel_x_prof, vel_y_prof, vel_z_prof, 
                        vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index

                vel_array.append(vel_z_prof[point_idx])
            
            if norm_val == -1:
                norm_val = max(vel_array)
            
            # plot
            if norm:
                ax1.plot(times, np.asarray(vel_array)/abs(min(vel_array)), lw=5, label="Point: ("+str(round(point[0]+self.x_off/1e3, 2))+","+str(round(point[1]+self.y_off/1e3, 2))+")")
            else:    
                ax1.plot(times, np.asarray(vel_array)*self.ms_to_mmyr*scaling_param, lw=5, label="Point: ("+str(round(point[0]+self.x_off/1e3, 2))+","+str(round(point[1]+self.y_off/1e3, 2))+")")

        # plot pressure function
        if pressure_func_file != None:
            df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            pressure_times = df['times']
            pressure_values = df['pressures']

            ax2 = ax1.twinx()
            ax2.set_ylabel(r"$P(t)/P_0$", fontsize=30)
            ax2.set_xlim([min(np.asarray(times)), max(np.asarray(times))])

            ax2.plot(pressure_times, np.asarray(pressure_values), linewidth=5, color='black', linestyle='dashed')
            ax2.tick_params(axis='y', labelsize=25)
        
        
        ax1.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=30)

        if pressure_func_file != None:
            ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
            ax1.patch.set_visible(False) # hide the 'canvas'

        # if norm:
        #     ax1.set_yscale('symlog')
        

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/points_time_"
            file_name += self.model_name.replace("/", "_") + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

    # Plot phase lag from different points over the pressurization history 
    def plot_phase_lag_time(self, times, points, plot_peaks=False, save=False):
        
         # set up plot
        plt.xlabel("Cycle", fontsize=self.label_fontsize)
        plt.ylabel("Phase Lag [rad]", fontsize=self.label_fontsize)

        plt.grid()
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)


        cycles = np.arange(self.model.cycles)
        central_vel_array = []

        # get center velocity array
        for t in times:
            model_time = self.get_timesteps(t)

            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time)
                
            point_idx = (np.abs(x_prof )).argmin() #get point index

            central_vel_array.append(vel_z_prof[point_idx])

        # time1_idx = (times <= self.model.spinup_time + self.model.T) & (times > self.model.spinup_time)
        # time2_idx = times > self.model.spinup_time + self.model.T

        # print(np.asarray(central_vel_array)[time1_idx])

        
        central_max_times_idx, _ = find_peaks(np.asarray(central_vel_array), distance=self.model.T-5)


        # iterate over points
        vel_array_list = []
        max_times_list = []
        for point in points:
            x = point[0]*1e3
            y = point[1]*1e3
            # convert x and y to r and theta
            r = np.sqrt(x**2 + y**2)

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y != 0:
                theta = np.pi/2
            else:
                theta = np.arctan(y/x)

            vel_array = []
            for t in times:
                model_time = self.get_timesteps(t)
                # print(t, model_time)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                        vel_x_prof, vel_y_prof, vel_z_prof, 
                        vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                
                point_idx = (np.abs(x_prof - r)).argmin() #get point index

                vel_array.append(vel_z_prof[point_idx])

            max_times_idx, _ = find_peaks(np.asarray(vel_array), distance=self.model.T-5)

            vel_array_list.append(vel_array)
            max_times_list.append(max_times_idx)
                
            # phase_lag = [(max_times[0]-central_max_times[0])*(2*np.pi/self.model.T),
            #             (max_times[1]-central_max_times[1])*(2*np.pi/self.model.T)]

            phase_lag = []
            for c in cycles:
                # print(times[max_times_idx[c]], times[central_max_times_idx[c]])
                phase_lag.append((times[max_times_idx[c]]-times[central_max_times_idx[c]])*(2*np.pi/self.model.T))
                   
            # plot
            plt.scatter(cycles+1, phase_lag, s=self.ms, label="Point: ("+str(point[0])+","+str(point[1])+")")

        plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=self.label_fontsize)

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/phase_lag_time"
            file_name += self.model_name.replace("/", "_") + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

        if plot_peaks:
            for i in range(len(vel_array_list)):
                plt.xlabel("Time [yrs]")
                plt.ylabel("Velocity [mm/yr]")
                plt.plot(times, np.asarray(central_vel_array)*self.ms_to_mmyr)
                plt.plot(times, np.asarray(vel_array_list[i])*self.ms_to_mmyr)
                plt.scatter(times[central_max_times_idx], np.asarray(central_vel_array)[central_max_times_idx]*self.ms_to_mmyr, marker="x")
                plt.scatter(times[max_times_list[i]], np.asarray(vel_array_list[i])[max_times_list[i]]*self.ms_to_mmyr, marker="x")
                plt.show()

    # Function that takes in a point and time span and returns the maximum phase lag between the point and central area
    def get_phaselag_center_point(self, times, point, debug=False):

        # convert point to r and theta
        x = point[0]*1e3
        y = point[1]*1e3
        r = np.sqrt(x**2 + y**2)

        if x == 0 and y == 0:
            theta = 0
        elif x == 0 and y != 0:
            theta = np.pi/2
        else:
            theta = np.arctan(y/x)

        center_vel = []
        outer_vel = []
        for t in times:
            center_vel.append(self.get_average_vel_area(t, (self.r_x, self.r_y))) # get velocity averaged over the source area

            model_time = self.get_timesteps(t)
                # print(t, model_time)

            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                
            point_idx = (np.abs(x_prof - r)).argmin() #get point index

            outer_vel.append(vel_z_prof[point_idx])

        central_max_times_idx, _ = find_peaks(-np.asarray(center_vel), distance=self.model.T-self.model.T/3)
        outer_max_times_idx, _ = find_peaks(-np.asarray(outer_vel), distance=self.model.T-self.model.T/3)
        print(len(central_max_times_idx), len(outer_max_times_idx))

        phase_lag = []
        for c in range(len(central_max_times_idx)):
            # print(times[outer_max_times_idx[c]], times[central_max_times_idx[c]])
            phase_lag.append((times[outer_max_times_idx[c]]-times[central_max_times_idx[c]])*(360/self.model.T)) # in degrees
        # print(phase_lag)

        if debug:
            plt.xlabel("Time [yrs]")
            plt.ylabel("Velocity [mm/yr]")
            plt.plot(times, np.asarray(center_vel))
            plt.plot(times, np.asarray(outer_vel)*self.ms_to_mmyr)
            plt.scatter(times[central_max_times_idx], np.asarray(center_vel)[central_max_times_idx], marker="x")
            plt.scatter(times[outer_max_times_idx], np.asarray(outer_vel)[outer_max_times_idx]*self.ms_to_mmyr, marker="x")
            plt.show()

        if len(phase_lag) != self.model.cycles:
            print("WARNING: num peaks found is different than the number of cycles.")
        return phase_lag
    
    # Function that takes in a point and time span and returns the cross correlation at lag 0 between the point velocity and the central area velocity
    def get_cc_center_point(self, times, point, debug=False):

         # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)

         # convert point to r and theta
        x = point[0]*1e3
        y = point[1]*1e3
        r = np.sqrt(x**2 + y**2)

        if x < 0:
            r *= -1
        if x == 0 and y == 0:
            theta = 0
        elif x == 0 and y > 0:
            theta = np.pi/2
        elif x == 0 and y < 0:
            theta = -np.pi/2
        else:
            theta = np.arctan(y/x)
        

        center_vel = []
        outer_vel = []
        for t in times:
            center_vel.append(self.get_average_vel_area(t, (self.r_x, self.r_y), offset=(self.x_off/1e3, self.y_off/1e3))) # get velocity averaged over the source area

            model_time = self.get_timesteps(t)
                # print(t, model_time)

            # (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
            #         vel_x_prof, vel_y_prof, vel_z_prof, 
            #         vel_r_prof, vel_theta_prof) = self.get_data_no_shift(model_time, theta=theta)
            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(model_time, theta=theta)

            # vel_z = self.get_point(model_time, x, y)
                
            point_idx = (np.abs(x_prof0 - r)).argmin() #get point index

            outer_vel.append(vel_z_prof[point_idx])
            # outer_vel.append(vel_z)


        center_vel_shift = np.asarray(center_vel)-np.mean(center_vel)
        outer_vel_shift = np.asarray(outer_vel)-np.mean(outer_vel)

        # cc_func = []
        # lags = np.arange(-2*self.model.T, 2*self.model.T, times[1]-times[0])
        # for tau in lags:
        #     cc = 0
        #     for t in times:
        #         if tau + t >= min(times) and tau+t <= max(times):
        #             # get index where of signals at time t
        #             t_idx =  (np.abs(times - t)).argmin()
        #             tau_t_idx = (np.abs(times - (tau+t))).argmin()

        #             cc+=center_vel_shift[t_idx]*outer_vel_shift[tau_t_idx]

        #     cc_func.append(cc)

        # cc_func = correlate(outer_vel_shift, center_vel_shift, method='auto')
        # lags = correlation_lags(len(center_vel_shift), len(outer_vel_shift))
        from scipy.stats import pearsonr
        result = pearsonr(outer_vel_shift, center_vel_shift)

        if debug:
            fig=plt.figure()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex = ax1)
            # ax3 = plt.subplot(313)
            ax1.plot(times, center_vel_shift)
            ax1.set_ylabel("Centered Center Vel [mm/yr]")
                
            ax2.plot(times, outer_vel_shift*self.ms_to_mmyr)
            ax2.set_ylabel("Centered Point Vel [mm/yr]")
            ax2.set_xlabel("Time [yrs]")

            # ax3.plot(lags, np.asarray(cc_func)/max(np.abs(cc_func)))
            # ax3.set_ylabel("Cross Correlation")
            # ax3.set_xlabel("Lags [yrs]")

            plt.show()

            plt.plot(times, center_vel/max(center_vel), lw=7, label='Mean Center Vel')
            plt.plot(times, np.asarray(outer_vel)/max(outer_vel), lw=6, label='Point ('+str(point[0])+","+str(point[1])+") Vel")

            plt.xlabel("time (yrs)", fontsize=30)
            plt.ylabel(r"Normalized $V_z$", fontsize=30)

            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(fontsize=30)
            plt.grid()
            plt.show()
            print(result[0])
        
        # return cc_func[(np.abs(lags)).argmin()]/max(np.abs(cc_func))
        return result[0]
    
    # Function that takes in a point and time span and returns the ratio between the maximum vel of the center average within the time span
    # and the subsidence vel of the point at that time, or None if the point isn't subsiding during max uplift
    def get_ratio_center_point(self, times, point, debug=False):

        # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)

        # convert point to r and theta
        x = point[0]*1e3
        y = point[1]*1e3
        r = np.sqrt(x**2 + y**2)

        if x < 0 and y < 0:
            r *= -1
        elif x < 0 and y >= 0:
            r *= -1

        if x == 0 and y == 0:
            theta = 0
        elif x == 0 and y != 0:
            theta = np.pi/2
        else:
            theta = np.arctan(y/x)

        center_vel = []
        outer_vel = []
        for t in times:
            center_vel.append(self.get_average_vel_area(t, (self.r_x, self.r_y))) # get velocity averaged over the source area

            model_time = self.get_timesteps(t)
                # print(t, model_time)

            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(model_time, theta=theta)
                
            point_idx = (np.abs(x_prof0 - r)).argmin() #get point index

            outer_vel.append(vel_z_prof[point_idx]*self.ms_to_mmyr)
        
        # get max uplift time at center
        # max_center_idx = np.argmax(center_vel)
        max_center_idx = np.argmax(center_vel)
        max_center_uplift = center_vel[max_center_idx]
        point_vel_max = outer_vel[max_center_idx]
        ratio_t_max = times[max_center_idx]

        min_center_idx = np.argmin(center_vel)
        min_center_uplift = center_vel[min_center_idx]
        point_vel_min = outer_vel[min_center_idx]
        ratio_t_min = times[min_center_idx]

        # if max_center_uplift <= 0 or point_vel >= 0:
        #     ratio = None
        # else:
        #     ratio = np.abs(max_center_uplift/point_vel)
        # if max_center_uplift * point_vel >= 0:
        #     ratio = None
        # else:
        #     ratio = np.abs(max_center_uplift/point_vel)

        ratio_max_center = max_center_uplift/point_vel_max
        ratio_min_center = min_center_uplift/point_vel_min

        if ratio_max_center < 0 and ratio_min_center > 0:
            ratio = np.abs(ratio_max_center)
            ratio_t = ratio_t_max
            point_vel = point_vel_max
        elif ratio_max_center > 0 and ratio_min_center < 0:
            ratio = np.abs(ratio_min_center)
            ratio_t = ratio_t_min
            point_vel = point_vel_min
        elif ratio_max_center < 0 and ratio_min_center < 0:
            if ratio_max_center > ratio_min_center:
                ratio = np.abs(ratio_max_center)
                ratio_t = ratio_t_max
                point_vel = point_vel_max
            else:
                ratio = np.abs(ratio_min_center)
                ratio_t = ratio_t_min
                point_vel = point_vel_min
        else:
            ratio = None
            ratio_t = ratio_t_max
            point_vel = point_vel_max

        if debug:
            # plot the center and point velocities and show where the ratio is being taken
            plt.grid()
            plt.xlabel("Time [yrs]")
            plt.ylabel("Velocity [mm/yr]")

            plt.plot(times, center_vel, label="Center")
            plt.plot(times, outer_vel, label="Point")
            plt.scatter(ratio_t, max_center_uplift, marker="x", color="black")
            plt.scatter(ratio_t, point_vel, marker="x", color="black")
            plt.legend()
            plt.show()

            plt.plot(times, center_vel/max(center_vel), lw=7, label='Mean Center Vel')
            plt.plot(times, np.asarray(outer_vel)/max(outer_vel), lw=6, label='Point ('+str(point[0])+","+str(point[1])+") Vel")

            if ratio != None:
                plt.scatter(ratio_t, max_center_uplift/max(center_vel), marker="x", color="black", s=300, zorder=10)
                plt.scatter(ratio_t, point_vel/max(outer_vel), marker="x", color="black", s=300, zorder=10)

            plt.xlabel("time (yrs)", fontsize=30)
            plt.ylabel(r"Normalized $V_z$", fontsize=30)

            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(fontsize=30)
            plt.grid()
            plt.show()

        return ratio
    
    # Function that plots all 0 lag CC's between the edge of the source and CR for a given time windoe
    def plot_cc_surface_time(self, times, CR_bounds, CR_shift=[0,0], inner_CR=None, inner_CR_shift=[0,0],
                              in_points_list=None, out_points_list=None, x_lim=None, y_lim=None, save=False,
                              CC_lim=[-1.0, 1.0]):

        # Create points arrays
        dx = 2.5
        x = np.arange(-CR_bounds[0]+CR_shift[0]-5, CR_bounds[0]+CR_shift[0]+dx, dx)
        y = np.arange(-CR_bounds[1]+CR_shift[1]-5, CR_bounds[1]+CR_shift[1]+dx, dx)

        # get CC's in region between source and CR
        # CC_list = np.empty(len(x)*len(y))
        # count = 0
        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         CC_list[count] = self.get_cc_center_point(times, (x[i], y[j]))

        #         count += 1
       
        # z = np.asarray(CC_list).reshape(len(x), len(y)).T

        CC_list = np.empty((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                CC_list[i][j] = self.get_cc_center_point(times, (x[i], y[j]))

        z = CC_list.T


        #plot
        plt.figure()
        #color_map = plt.cm.get_cmap('RdBu').reversed()
        # color_map='seismic'
        color_map='Spectral_r'

        plt.imshow(z, extent=(-CR_bounds[0]+CR_shift[0]-5,CR_bounds[0]+CR_shift[1],-CR_bounds[1]+CR_shift[1]-5,CR_bounds[1]+CR_shift[1]),cmap=color_map, interpolation="nearest", origin="lower",
                   vmax=CC_lim[1], vmin=CC_lim[0])
        # plt.imshow(z, extent=(-CR_bounds[0],CR_bounds[0],-CR_bounds[1],CR_bounds[1]),cmap=color_map, interpolation="nearest", origin="lower")
        plt.xlabel("X [km]", fontsize=self.label_fontsize)
        plt.ylabel("Y [km]", fontsize=self.label_fontsize)
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="CC", size=self.label_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_fontsize)
        
        # plot CR
        CR_x_points, CR_y_points = get_CR_points(CR_bounds[0], CR_bounds[1])
        plt.plot(CR_x_points+CR_shift[0], CR_y_points+CR_shift[1], c='black', linestyle='dashed')

        # plot inner CR if requested
        if inner_CR != None:
            inner_CR_x_points, inner_CR_y_points = get_CR_points(inner_CR[0], inner_CR[1])
            plt.plot(inner_CR_x_points+inner_CR_shift[0], inner_CR_y_points+inner_CR_shift[1], c='black', linestyle='dashdot')

        # plot source
        source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
        plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot')

        if in_points_list != None:
            for point in in_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='indigo', marker='s', s=200, edgecolor='black', linewidths=2, zorder=2)
        if out_points_list != None:
            for point in out_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='teal', marker='s', s=200, edgecolor='black', linewidths=2)

        # set plot limits
        if x_lim != None:
            plt.xlim(x_lim)
        if y_lim != None:
            plt.ylim(y_lim)

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/CC_surface"
            file_name += self.model_name.replace("/", "_") + "_" + str(times[0])+"-"+str(times[len(times)-1])+".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()
          
    # Function that plots all ratios between the edge of the source and CR for a given time window
    def plot_ratio_surface_time(self, times, CR_bounds, CR_shift=[0,0], inner_CR=None, inner_CR_shift=[0,0],
                                 in_points_list=None, out_points_list=None, x_lim=None, y_lim=None, save=False, ratio_lim=[0.5, 40]):

        # Create points arrays
        # dx = 5
        dx = 2.5
        x = np.arange(-CR_bounds[0]+CR_shift[0], CR_bounds[0]+CR_shift[0]+dx, dx)
        y = np.arange(-CR_bounds[1]+CR_shift[1], CR_bounds[1]+CR_shift[1]+dx, dx)


        # get CC's in region between source and CR
        ratio_list = np.empty(len(x)*len(y))
        count = 0
        for i in range(len(x)):
            for j in range(len(y)):
                ratio_list[count] = self.get_ratio_center_point(times, (x[i], y[j]))
                count += 1
        print("Done ratios")

        z = np.asarray(ratio_list).reshape(len(x), len(y)).T

        #plot
        plt.figure()
        #color_map = plt.cm.get_cmap('RdBu').reversed()
        # color_map='seismic'
        color_map='cool'

        # plt.imshow(z, extent=(-CR_bounds[0],CR_bounds[0],-CR_bounds[1],CR_bounds[1]),cmap=color_map, norm=matplotlib.colors.LogNorm(), interpolation="nearest")
        plt.imshow(z, extent=(-CR_bounds[0]+CR_shift[0],CR_bounds[0]+CR_shift[0],-CR_bounds[1]+CR_shift[1],CR_bounds[1]+CR_shift[1]),cmap=color_map, vmin=ratio_lim[0], vmax=ratio_lim[1], interpolation="nearest", origin="lower")
        plt.xlabel("X [km]", fontsize=self.label_fontsize)
        plt.ylabel("Y [km]", fontsize=self.label_fontsize)
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Uplift/Subsidence Ratio", size=self.label_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_fontsize)

        # plot CR
        CR_x_points, CR_y_points = get_CR_points(CR_bounds[0], CR_bounds[1])
        plt.plot(CR_x_points+CR_shift[0], CR_y_points+CR_shift[1], c='black', linestyle='dashed')

        # plot inner CR if requested
        if inner_CR != None:
            inner_CR_x_points, inner_CR_y_points = get_CR_points(inner_CR[0], inner_CR[1])
            plt.plot(inner_CR_x_points+inner_CR_shift[0], inner_CR_y_points+inner_CR_shift[1], c='black', linestyle='dashdot')

        # plot source
        source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
        plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot')

        if in_points_list != None:
            for point in in_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='blue', marker='s', s=200, edgecolor='black', linewidths=2, zorder=2)
        if out_points_list != None:
            for point in out_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='orange', marker='s', s=200, edgecolor='black', linewidths=2)

        # set plot limits
        if x_lim != None:
            plt.xlim(x_lim)
        if y_lim != None:
            plt.ylim(y_lim)

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/ratio_surface"
            file_name += self.model_name.replace("/", "_") + "_" + str(times[0])+"-"+str(times[len(times)-1])+".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()


    # Function to plot the velocity of the central point and arbitrary outer points over time
    # points should be a list of outer points only
    def plot_center_outer_ratio(self, times, points, save=False):

        # set up plot
        plt.xlabel("Time [years]", fontsize=self.label_fontsize)
        plt.ylabel("Velocity Ratio", fontsize=self.label_fontsize)
        plt.grid()
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)

        center_vel = []
        for t in times:
            model_time = self.get_timesteps(t)

            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                        vel_x_prof, vel_y_prof, vel_z_prof, 
                        vel_r_prof, vel_theta_prof) = self.get_data(model_time)
                
            point_idx = (np.abs(x_prof - 0)).argmin() #get point index
            center_vel.append(vel_z_prof[point_idx])
            
        for point in points:
            x = point[0]*1e3
            y = point[1]*1e3
            # convert x and y to r and theta
            r = np.sqrt(x**2 + y**2)

            if x == 0:
                theta = 0
            elif x == 0 and y != 0:
                theta = np.pi/2
            else:
                theta = np.arctan(y/x)

            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)
                # print(t, model_time)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                        vel_x_prof, vel_y_prof, vel_z_prof, 
                        vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                
                point_idx = (np.abs(x_prof - r)).argmin() #get point index

                vel_array.append(vel_z_prof[point_idx])
            
            plt.plot(times, np.asarray(center_vel)/np.asarray(vel_array), lw=self.lw, label="Point: ("+str(point[0])+","+str(point[1])+")")
        
        plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=self.label_fontsize)

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/center_outer_ratio"
            file_name += self.model_name.replace("/", "_") + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

    
    # get model characteristic time, update csv and plot
    def get_tc(self, plot=True, save_plot=False, plot_title=None, save_data=False, save_directory=None):
        
        #time steps standardized for all tc calculations
        time_steps = np.asarray([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90])

        #collect max vz
        vz_max = []
        for t in time_steps:
            model_time = self.get_timesteps(t)

            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time)

            vz_max.append(max(vel_z))

        #get log of vel data in mm/yr
        vz_max_log = np.log(np.asarray(vz_max)*self.ms_to_mmyr) 

        p = np.polyfit(time_steps[0:5], vz_max_log[0:5], 1)
        fit = np.exp(p[0]*time_steps + p[1])

        tc = 1.0/np.abs(p[0])

        #write tc to csv
        self.model.update_csv(tc=tc)
        print(tc)

        if plot or save_data:
            plt.grid()
            plt.xlabel("Time [years]", fontsize=self.label_fontsize)
            plt.ylabel(r" $v_z$(x=0, y=0) [mm/yr]", fontsize=self.label_fontsize)
            plt.yscale("log")
            plt.xticks(fontsize=self.axes_fontsize)
            plt.yticks(fontsize=self.axes_fontsize)

            plt.scatter(time_steps, np.asarray(vz_max)*self.ms_to_mmyr, label="data", s=self.ms)
            plt.plot(time_steps, np.asarray(fit), label="fit", color='orange', linewidth=self.lw, linestyle="dashed")

            if plot_title != None:
                plt.title(plot_title, fontsize=self.title_fontsize)

            if save_plot:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/get_tc"
                file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.tr) + ".png"
                plt.savefig(file_name, bbox_inches="tight")

        if save_data:
            #create a run directory if it doesn't exist
            directory = "/home/grantblock/Research/SMBPylith/RunData/"+save_directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            #save a figure in that directory
            fig_name = directory + "CharTimes_fig_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(fig_name, bbox_inches="tight")    
            #save the csv
            csv_name = directory + "CharTimes_csv_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".csv"
            column_names = ["Time Steps [yrs]", "V_z(0) [mm/yr]", "V_z fit [mm/yr]"]
            column_list = [time_steps, np.asarray(vz_max)*self.ms_to_mmyr, np.asarray(fit)]
            self.save_csv(column_list, column_names, csv_name)
            
        if plot or save_data:
            plt.show()
    
    # get center and shoulder velocities and use them to find the sombrero duration.
    # can plot the center shoulder velocities as well
    def get_som_dur(self, time_steps, plot=True, use_ry=False, theta=None, save_plot=False, plot_title=None, save_data=False, save_directory=None, verbose=True):
        
        #set the shoulder
        if use_ry:
            r_source = self.r_y
        else:
            r_source=self.r_x

        shoulder = r_source*1e3*self.shoulder_ratio

        #get x_0 
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        #initialize lists to collect center and shoulder vz
        vz_center = []
        vz_shoulder = []
    
        somb_durations = []
        current_somb = False
   
        first = True

        for t in time_steps:

            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            x = x_0
            
            if first:
                center_idx = (np.abs(x - 0.0)).argmin() #get center index
                shoulder_idx = (np.abs(x - shoulder)).argmin() #get shoulder index
                first = False
            
            vz_center.append(vel_z[center_idx])
            vz_shoulder.append(vel_z[shoulder_idx])
                      
            if (vel_z[center_idx] > 0) and (vel_z[shoulder_idx] < 0):
                if not current_somb:
                    t0 = t
                    current_somb = True
                if verbose:
                    print("Sombrero at ", t)
                somb_time = t
            else:
                if current_somb:
                    current_somb = False
                    somb_durations.append(somb_time-t0)
                    
        if verbose:
            if len(somb_durations) == 0:
                print("Sombrero Durations = 0")
            else:
                print("Sombrero Durations = ", somb_durations)

        #write t_som to csv
        self.model.update_csv(t_som=somb_durations)
        
        #plot data
        if plot or save_data:
            plt.xlabel("Time [years]", fontsize=self.label_fontsize)
            plt.ylabel("Velocity [mm/yr]", fontsize=self.label_fontsize)
    
            #plt.scatter(np.asarray(time_steps), np.asarray(vz_center)*self.ms_to_mmyr, label=r"Center $v_z$", s=self.ms)
            plt.plot(np.asarray(time_steps), np.asarray(vz_center)*self.ms_to_mmyr, linewidth=self.lw, label=r"Center $v_z$")
        
            #plt.scatter(np.asarray(time_steps), np.asarray(vz_shoulder)*self.ms_to_mmyr, label=r"Shoulder $v_z$", s=self.ms)
            plt.plot(np.asarray(time_steps), np.asarray(vz_shoulder)*self.ms_to_mmyr, linewidth=self.lw, label=r"Shoulder $v_z$")
        
            if plot_title != None:
                plt.title(plot_title + r", $v_z$", fontsize=20)
          
            plt.grid()
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.label_fontsize)
            plt.xticks(fontsize=self.tick_fontsize)
            plt.yticks(fontsize=self.tick_fontsize)
        
            if save_plot:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/center_shoulder_vz_"
                file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
                plt.savefig(file_name, bbox_inches="tight")
                
        if save_data:
            #create a run directory if it doesn't exist
            directory = "/home/grantblock/Research/SMBPylith/RunData/"+save_directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            #save a figure in that directory
            fig_name = directory + "CenterShoulder_zoom_fig_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(fig_name, bbox_inches="tight")    
            #save the csv
            csv_name = directory + "CenterShoulder_zoom_csv_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".csv"
            column_names = ["Time Steps [yrs]", "Center V_z [mm/yr]", "Shoulder V_z [mm/yr]"]
            column_list = [np.asarray(time_steps), np.asarray(vz_center)*self.ms_to_mmyr, np.asarray(vz_shoulder)*self.ms_to_mmyr]
            self.save_csv(column_list, column_names, csv_name)
            
        if plot or save_data:
            plt.show()

        return somb_durations
    
    # a function to find and plot sombrero magnitudes
    def get_som_mag(self, time_steps, plot=True, use_ry=False, theta=None, save_plot=False, plot_title=None, save_data=False, save_directory=None, verbose=True):
        
        #set the shoulder
        if use_ry:
            r_source = self.r_y
        else:
            r_source = self.r_x

        shoulder = r_source*1e3*self.shoulder_ratio

        #collect center and shoulder vz
        vz_center = []
        vz_shoulder = [] 
        
        first = True
        for t in time_steps:

            model_time = self.get_timesteps(t)
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            
            if first:
                center_idx = (np.abs(x - 0.0)).argmin() #get center index
                shoulder_idx = (np.abs(x - shoulder)).argmin() #get shoulder index
                first = False
            
            vz_center.append(vel_z[center_idx])
            vz_shoulder.append(vel_z[shoulder_idx])

        #make a list of sombrero magnitudes
        somb_mags = []
    
        for i in range(len(vz_center)):
            if vz_center[i] > 0 and vz_shoulder[i] < 0:
                somb_mags.append(np.abs(vz_shoulder[i])/vz_center[i])
            else:
                somb_mags.append(0.0)
        #find peaks
        peaks, prop = find_peaks(somb_mags, height=1e-2)
        if verbose:
            print(prop['peak_heights'])

        #write tc to csv
        self.model.update_csv(som_mag=list(prop['peak_heights']))


        #plot the sombrero magnitude
        if plot or save_data:
            plt.xlabel("Time [years]", fontsize=self.label_fontsize)
            plt.ylabel("Sombrero Magnitude", fontsize=self.label_fontsize)
        
            plt.scatter(np.asarray(time_steps), np.asarray(somb_mags), s=self.ms)
            plt.plot(np.asarray(time_steps), np.asarray(somb_mags), linewidth=self.lw)
        
            if plot_title != None:
                plt.title(plot_title, fontsize=20)
            plt.grid()
            plt.xticks(fontsize=self.tick_fontsize)
            plt.yticks(fontsize=self.tick_fontsize)
            
        if save_data:
            directory = "/home/grantblock/Research/SMBPylith/RunData/"+save_directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            #save a figure in that directory
            fig_name = directory + "SombreroMag_fig_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(fig_name, bbox_inches="tight")    
            #save the csv
            csv_name = directory + "SombreroMag_csv_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".csv" 
            column_names = ["Time Steps [yrs]", "Sombrero Magnitude"]
            column_list = [np.asarray(time_steps), np.asarray(somb_mags)]
            self.save_csv(column_list, column_names, csv_name)
            supp_name = directory + "SombMags_supplimentary" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".txt"
        
        if plot or save_data:
            plt.show()

    # find Delta V and plot it
    def get_delta_v(self, time_steps, bounds, use_ry=False, theta=None, exp=False, plot=True, save_plot=False, plot_title=None, save_data=False, save_directory=None, verbose=True):
        
        #set the shoulder
        if use_ry:
            r_source=self.r_y
        else:
            r_source=self.r_x

        shoulder = r_source*1e3*self.shoulder_ratio

        #collect center and shoulder vz
        vz_center = []
        
        first = True
        for t in time_steps:

            model_time = self.get_timesteps(t)
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            
            if first:
                center_idx = (np.abs(x - 0.0)).argmin() #get center index
                first = False
            
            vz_center.append(vel_z[center_idx])

        vz_center_init = vz_center[0]

        #remove periodic parts from velocity function to fit exponential curve
        periodic_window_idx = (np.asarray(time_steps) >= bounds[0]) & (np.asarray(time_steps) <= bounds[1])

        #fit function
        non_periodic_timesteps = np.asarray(time_steps)[~periodic_window_idx]
        non_periodic_center = np.asarray(vz_center)[~periodic_window_idx]
                
        #fit exponential function
        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c
            
        def quad_log_func(x, a, b, c):
            return a + b*np.log(x) + c*np.log(x)**2
        
        if exp:
            func = exp_func
            func_name = "Exponential Fit"
        else:
            func = quad_log_func
            func_name = "Quad Log Fit"
        center_popt, center_pcov = curve_fit(func, non_periodic_timesteps, non_periodic_center*self.ms_to_mmyr, p0=[1., 1., vz_center[len(vz_center)-1]])

        #get fit errors
        model_predictions = func(non_periodic_timesteps, *center_popt)
        abs_error = model_predictions - np.asarray(non_periodic_center)*self.ms_to_mmyr
        SE = np.square(abs_error) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        
        center_RMSE=RMSE
        r_squared = 1.0 - (np.var(abs_error) / np.var(np.asarray(non_periodic_center)*self.ms_to_mmyr))
        center_r_squared=r_squared
        if verbose:
            print("Center RMSE:", str(RMSE), "Center R Square:", str(r_squared))

        #FIND DELTA_V
    
        #find for center
        #get residual
        res = np.asarray(vz_center)*self.ms_to_mmyr - func(time_steps, *center_popt)
        
        #get time of the max of the "wiggles". We exclude the first non-periodic part in case that is higher than the wiggles
        post_init_relax_idx = np.asarray(time_steps) > bounds[0] #get array after initial relaxation
        init_relax_idx = np.asarray(time_steps) < bounds[0] #get array during
        
        #sort res from greatest to least    
        sorted_res = sorted(res[post_init_relax_idx],reverse=True)
        sorted_idx = [list(res).index(i) for i in sorted_res]
        
        max_vz1_idx = sorted_idx[0]
        max_vz2_idx = np.inf #just so we get an error if its not assigned
        for i in sorted_idx:
            if (time_steps[i] - time_steps[max_vz1_idx] > self.model.T/1.5 or time_steps[max_vz1_idx] - time_steps[i] > self.model.T/1.5) and res[i] > 0:
                max_vz2_idx = i
                break
            
        
        max_time1_center = time_steps[max_vz1_idx]
        max_vz1_center = vz_center[max_vz1_idx]*self.ms_to_mmyr

        max_time2_center = time_steps[max_vz2_idx]
        max_vz2_center = vz_center[max_vz2_idx]*self.ms_to_mmyr

        if verbose:
            print("Center Max Vz's 1 and 2:", str(max_vz1_center), str(max_vz2_center))

        fit_at_max1_center = func(max_time1_center, *center_popt) #find the value of the fit at the time the wiggle is at a max, then subtract to get delta v
        fit_at_max2_center = func(max_time2_center, *center_popt)
        
        
        delta_v_center = ((max_vz1_center - fit_at_max1_center) + (max_vz2_center - fit_at_max2_center))/2 #delta v is the average between the two peaks
        if verbose:
            print("Center Delta_V:", str(delta_v_center), "mm/yr")

        #write t_som to csv
        self.model.update_csv(Delta_v=delta_v_center)

        #get v_in
        v_in_center = np.asarray(vz_center)[init_relax_idx][len(np.asarray(vz_center)[init_relax_idx])-1]*self.ms_to_mmyr
        if verbose:
            print("Center v_in", str(v_in_center), "mm/yr")

        if plot or save_data:
            plt.scatter(np.asarray(time_steps), np.asarray(vz_center)*self.ms_to_mmyr, s=self.ms, color="blue", label=r"Center $v_z$")
            plt.plot(np.asarray(time_steps), func(time_steps, *center_popt), color="orange", label=func_name, linewidth=self.lw)
            plt.scatter(np.asarray(non_periodic_timesteps), non_periodic_center*self.ms_to_mmyr, s=self.ms, color="red", label=r"Non-Periodic Center $v_z$")
            plt.plot(np.asarray([max_time1_center, max_time1_center]), [fit_at_max1_center, max_vz1_center], linewidth=3, color='black', label=r"$\Delta v_z$")
            plt.plot(np.asarray([max_time2_center, max_time2_center]), [fit_at_max2_center, max_vz2_center], linewidth=3, color='black')
            

            plt.grid()
            if plot_title != None:
                plt.title(plot_title, fontsize=self.title_fontsize)
            plt.xlabel("Time [years]", fontsize=self.label_fontsize)
            plt.ylabel("Velocity [mm/yr]", fontsize=self.label_fontsize)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.label_fontsize)
            plt.xticks(fontsize=self.tick_fontsize)
            plt.yticks(fontsize=self.tick_fontsize)

            if save_plot:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/delta_v_"
                file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
                plt.savefig(file_name, bbox_inches="tight")

        if save_data:

            #create a run directory if it doesn't exist
            directory = "/home/grantblock/Research/SMBPylith/RunData/"+save_directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            #save a figure in that directory
            fig_name = directory + "DeltaV_fig_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(fig_name, bbox_inches="tight")
            plt.show()
            #save the csv
            csv_name = directory + "DeltaV_csv_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".csv"
            column_names = ["Time Steps [yrs]", "Center V_z [mm/yr]", func_name + " [mm/yr]", "Non Periodic Time Steps [yrs]", "Non Periodic Center V_z [mm/yr]"]
            column_list = [np.asarray(time_steps), np.asarray(vz_center)*self.ms_to_mmyr, func(time_steps, *center_popt), np.asarray(non_periodic_timesteps), np.asarray(non_periodic_center)*self.ms_to_mmyr]
            self.save_csv(column_list, column_names, csv_name)
            #write another file with extra data
            supp_name = directory + "DeltaV_supplimentary" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".txt"
            f = open(supp_name, "a")
            f.write("Center Delta V = "+str(delta_v_center)+" mm/yr\n")
            f.write("Center Delta V 1 time = " + str(max_time1_center*self.model.output_dt)+" yr\n")
            f.write("Center Delta V 2 time = " + str(max_time2_center*self.model.output_dt)+" yr\n")
            f.write("Fit RMSE = " + str(center_RMSE)+"\n")
            f.write("Fit r^2 = " + str(center_r_squared)+"\n")
            f.close()

        if plot or save_data:    
            plt.show()

    
    # make center shoulder phase lag plot
    def phase_lag_plot(self, time_steps=[], use_ry=False, theta=None, save=False, title=None, save_data=False, save_directory=None, dashed=False):

        #set the shoulder
        if use_ry:
            r_source = self.r_y
        else:
            r_source = self.r_x

        shoulder = r_source*1e3*self.shoulder_ratio
        
        #get x_0 
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        #set time steps for this plot
        if len(time_steps) == 0:
            time_steps = np.arange(self.model.spinup_time, self.model.spinup_time+self.model.cycles*self.model.T, 1).astype(int)
        
        #collect center and shoulder vz
        vz_center = []
        vz_shoulder = [] 
        
        first = True
        for t in time_steps:

            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            x = x_0
            if first:
                center_idx = (np.abs(x - 0.0)).argmin() #get center index
                shoulder_idx = (np.abs(x - shoulder)).argmin() #get shoulder index
                first = False
            
            vz_center.append(vel_z[center_idx])
            vz_shoulder.append(vel_z[shoulder_idx])
        
        #make the pressure function
        if type(self.model.output_dt) == list:
            dt = self.model.output_dt[1]
        else:
            dt = self.model.output_dt
        times = np.arange(0.0, self.model.run_time+2*dt, dt) #times should end 1 dt after total simulation time for pylith to be happy
        A = 1
        if self.model.sawtooth:
            T_inc_percent=0.25
            T_dec_percent=0.75
        elif self.model.inv_sawtooth:
            T_inc_percent=0.75
            T_dec_percent=0.25
        
        if self.model.sawtooth or self.model.inv_sawtooth:

            pressures = []
            completed_cycles = 0
            T_inc = T_inc_percent*self.model.T
            T_dec = T_dec_percent*self.model.T
            for t in times:
                m_inc = A/T_inc
                m_dec = -A/T_dec
                if (t > self.model.spinup_time + completed_cycles*self.model.T) and (t <= self.model.spinup_time + completed_cycles*self.model.T + T_inc) and (completed_cycles < self.model.cycles):
                    pressures.append(1.0+m_inc*(t-(self.model.spinup_time + completed_cycles*self.model.T)))
                elif (t >= self.model.spinup_time + completed_cycles*self.model.T + T_inc) and (t < self.model.spinup_time + (1+completed_cycles)*self.model.T) and (completed_cycles < self.model.cycles):
                    pressures.append(1.0+A+m_dec*(t-(self.model.spinup_time + completed_cycles*self.model.T + T_inc)))
                else:
                    pressures.append(1.0)
    
                if t > 0 and t == self.model.spinup_time + (completed_cycles+1)*self.model.T:
                    completed_cycles+=1
        else:
            sine = -A*np.sin((2*np.pi/self.model.T)*(times-self.model.spinup_time)) + 1

            pressures = np.ones(len(times))

            sine_times = np.where((times > self.model.spinup_time) & (times <= self.model.spinup_time + self.model.cycles*self.model.T))
            pressures[sine_times] = sine[sine_times]
        
        #plot
        fig, ax1 = plt.subplots()
        
        color='tab:blue'
        ax1.set_xlabel("Time [years]", fontsize=self.label_fontsize)
        ax1.set_ylabel("Normalized Velocity", fontsize=self.label_fontsize, color=color)
    
        ax1.plot(np.asarray(time_steps), np.asarray(vz_center)/np.max(np.abs(vz_center)), linewidth=self.lw, label=r"Center $v_z$")
        
        if dashed:
            ax1.plot(np.asarray(time_steps), np.asarray(vz_shoulder)/np.max(np.abs(vz_shoulder)), linewidth=self.lw, label=r"Shoulder $v_z$", linestyle="dashed")
        else:
            ax1.plot(np.asarray(time_steps), np.asarray(vz_shoulder)/np.max(np.abs(vz_shoulder)), linewidth=self.lw, label=r"Shoulder $v_z$")
        
        ax1.tick_params(axis='y', labelcolor=color, labelsize=self.tick_fontsize)
        ax1.tick_params(axis='x', labelsize=self.tick_fontsize)
        
        ax1.grid()

        
        ax2 = ax1.twinx()
        color='black'
        ax2.set_ylabel("Pressure Function Amplitude", fontsize=self.label_fontsize)
        ax2.set_xlim([min(np.asarray(time_steps)), max(np.asarray(time_steps))])
        ax2.plot(times, np.asarray(pressures)-1, linewidth=self.lw, color=color, linestyle='dashed', label="Pressure Function")
        ax2.tick_params(axis='y', labelcolor=color, labelsize=self.axes_fontsize)
        fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.tick_fontsize)

        fig.tight_layout()
        #plt.grid()
        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/center_shoulder_vz_pl_"
            file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(file_name, bbox_inches="tight")
            
        if save_data:
            #create a run directory if it doesn't exist
            directory = "/home/grantblock/Research/SMBPylith/RunData/"+save_directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            #save a figure in that directory
            fig_name = directory + "CenterShoulder_full_fig_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(fig_name, bbox_inches="tight")    
            #save the csv
            pressures_arr = np.asarray(pressures)
            pressures_cut = pressures_arr[time_steps] - 1.0
            csv_name = directory + "CenterShoulder_full_csv_" + self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".csv"
            column_names = ["Time Steps [yrs]", "Normalized Center V_z", "Normalized Shoulder V_z", "Normalized Pressure Function"]
            column_list = [np.asarray(time_steps)*dt, np.asarray(vz_center)/np.abs(np.max(vz_center)), np.asarray(vz_shoulder)/np.abs(np.max(vz_shoulder)), pressures_cut]
            self.save_csv(column_list, column_names, csv_name)
        plt.show()

    # center shoulder phase lag plot with multiple pressurizing sources. Takes a list of timedbs to plot the multiple pressure functions
    def phase_lag_plot_multisource(self, timedb_list, time_steps=[], use_ry=False, theta=None, save=False, title=None, dashed=False):
        #set the shoulder
        if use_ry:
            r_source = self.r_y
        else:
            r_source = self.r_x

        shoulder = r_source*1e3*self.shoulder_ratio

        #get x_0 
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)


        #set time steps for this plot
        if len(time_steps) == 0:
            time_steps = np.arange(self.model.spinup_time, self.model.spinup_time+self.model.cycles*self.model.T, 1)
        
        #collect center and shoulder vz
        vz_center = []
        vz_shoulder = [] 
        
        first = True
        for t in time_steps:

            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
            x = x_0
            if first:
                center_idx = (np.abs(x - 0.0)).argmin() #get center index
                shoulder_idx = (np.abs(x - shoulder)).argmin() #get shoulder index
                first = False
            
            vz_center.append(vel_z[center_idx])
            vz_shoulder.append(vel_z[shoulder_idx])

        #read the timedbs
        timedb_times = []
        timedb_pressures = []
        timedb_linestyle_list = ['dashed', 'dotted', 'dashdot']
        for name in timedb_list:
            df = pd.read_csv(name, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            timedb_times.append(df['times'])
            timedb_pressures.append(df['pressures'])

        # plot center, shoulder and pressures
        fig, ax1 = plt.subplots()
        
        color='tab:blue'
        ax1.set_xlabel("Time [years]", fontsize=self.label_fontsize)
        ax1.set_ylabel("Normalized Velocity", fontsize=self.label_fontsize, color=color)
    
        ax1.plot(np.asarray(time_steps), np.asarray(vz_center)/np.max(np.abs(vz_center)), linewidth=self.lw, label=r"Center $v_z$")
        
        if dashed:
            ax1.plot(np.asarray(time_steps), np.asarray(vz_shoulder)/np.max(np.abs(vz_shoulder)), linewidth=self.lw, label=r"Shoulder $v_z$", linestyle="dashed")
        else:
            ax1.plot(np.asarray(time_steps), np.asarray(vz_shoulder)/np.max(np.abs(vz_shoulder)), linewidth=self.lw, label=r"Shoulder $v_z$")
        
        ax1.tick_params(axis='y', labelcolor=color, labelsize=self.tick_fontsize)
        ax1.tick_params(axis='x', labelsize=self.tick_fontsize)
        
        ax1.grid()

        ax2 = ax1.twinx()
        color='black'
        ax2.set_ylabel("Pressure Function Amplitude Change", fontsize=self.label_fontsize)
        ax2.set_xlim([min(np.asarray(time_steps)), max(np.asarray(time_steps))])
        for i in range(len(timedb_times)):
            ax2.plot(timedb_times[i], np.asarray(timedb_pressures[i])-1, linewidth=self.lw, color=color, linestyle=timedb_linestyle_list[i], label="Pressure Function "+str(i))
        ax2.tick_params(axis='y', labelcolor=color, labelsize=self.axes_fontsize)
        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.tick_fontsize)
        fig.tight_layout()
        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/center_shoulder_vz_pl_"
            file_name += self.model_name.replace("/", "_") + "_relax=" + str(self.model.tr) + ".png"
            plt.savefig(file_name, bbox_inches="tight")
            
        plt.show()


    # read in the full .h5 at a given time step and find the volume change from the pressurizing source
    def get_ellipsoid_volume(self, t, axis_z=1.875e3, depth=6.5e3):

        #convert to model time
        model_t = self.get_timesteps(t)
        
        #Read in the source cylinder .h5
        #SMB_chamber_bot_50km-1000-1_yr_relax.h5
        if "Yellowstone" in self.model.model_name: #different parsing strategy for yellowstone models
            path = "../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+".h5"
        elif self.model.tr == 0.1:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_01_yr_relax.h5"
        else:
            path = "../../"+str(self.model.path)+"/output/SMB_chamber_bot_50km-groundsurf-" + str(self.model.run_time)+"_"+str(int(self.model.tr))+"_yr_relax.h5" 

        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            z = points[:][:,2]
            
            disp_x = displacements[model_t][:][:,0]
            disp_y = displacements[model_t][:][:,1]
            disp_z = displacements[model_t][:][:,2]

            #track points on the three axes of the ellipse
            axis_x = self.r_x*1e3
            axis_y = self.r_y*1e3
            # tol_x = 7e3 #minimum mesh distance on the source
            # tol_y = 7e3
            # tol_z = 7e3

            # mask_x = (np.abs(x - axis_x) < tol_x)  & (np.abs(y - 0.0) < tol_y) & (np.abs((z+depth)-0.0) < tol_z)
            # mask_y = (np.abs(x - 0.0) < tol_x)  & (np.abs(y - axis_y) < tol_y) & (np.abs((z+depth)-0.0) < tol_z)
            # mask_z = (np.abs(x - 0.0) < tol_x)  & (np.abs(y - 0.0) < tol_y) & (np.abs((z+depth)-axis_z) < tol_z)

            # print(np.count_nonzero(mask_x==True), np.count_nonzero(mask_y==True), np.count_nonzero(mask_z==True))

            mask_x = np.argmin(np.abs(x - axis_x).all() and np.abs(y).all() and np.abs((z+depth)).all())
            mask_y = np.argmin(np.abs(x).all() and np.abs(y - axis_y).all() and np.abs((z+depth)).all())
            mask_z = np.argmin(np.abs(x).all() and np.abs(y).all() and np.abs((z+depth)-axis_z))
            
            #get the disp at the axes in the direction of the axes
            disp_x_axis_x = disp_x[mask_x]
            disp_y_axis_y = disp_y[mask_y]
            disp_z_axis_z = disp_z[mask_z]
            
            #get the volume of the ellipsoid by adding the displacements to the axes
            volume = (4./3.)*np.pi*(axis_x+disp_x_axis_x)*(axis_y+disp_y_axis_y)*(axis_z+disp_z_axis_z) #in m^3
            return volume
        
    
    # A function to plot sombrero width as a function of time through sombrero
    def somb_width_time(self, time_steps, angle_avg=True, mesh_size=False, save=False, save_name=None):

        # list of angles to average over
        #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
        angles = [0, np.pi/2.]


        # collect sombrero widths at each time step
        width_list = []
        mesh_size_0 = []
        mesh_size_90 = []
        mesh_size_neg = []
        for t in time_steps:
            model_time = self.get_timesteps(t)
            vel_z_list = []
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time)
            x0=x
            if angle_avg:
                vel_z_list.append(vel_z)
                #average through the different angles
                for theta in angles:
                    (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)
                    vel_z_list.append(vel_z) 
                vel_z_final = np.mean(np.asarray(vel_z_list), axis=0)
            else:
                vel_z_final = vel_z
            
            #get width
            width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x0), np.asarray(vel_z_final)*self.ms_to_mmyr)
            
            if mesh_size:
                mesh_size_0.append(self.get_mesh_size(pos_min_x, 0))
                # mesh_size_90.append(self.get_mesh_size(pos_min_x, np.pi/2.))
                # mesh_size_neg.append(self.get_mesh_size(neg_min_x, 0))

            #width_list.append(width)
            width_list.append(2*pos_min_x)

        #make plot
        if mesh_size:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ax1.scatter(time_steps-time_steps[0], np.asarray(width_list)/1e3, c="blue")
            ax2.scatter(time_steps-time_steps[0], np.asarray(mesh_size_0)/1e3, s=self.ms-40, c="orange", label="Mesh Size at 0$^\circ$")
            # ax2.scatter(time_steps-time_steps[0], np.asarray(mesh_size_90)/1e3, s=self.ms-60, c="green", marker='s', label="Mesh Size at 90$^\circ$")
            # ax2.scatter(time_steps-time_steps[0], np.asarray(mesh_size_neg)/1e3, s=self.ms-60, c="purple", marker='+', label="Mesh Size at negative trough")


            ax1.set_xlabel("Time After Start of Sombrero [yrs]", fontsize=self.label_fontsize)
            ax1.set_ylabel("Sombrero Width [km]", fontsize=self.label_fontsize, color="blue")
            ax2.set_ylabel("Mesh Size [km]", fontsize=self.label_fontsize, color="orange")

            ax1.tick_params(axis='y', labelcolor="blue", labelsize=self.axes_fontsize)
            ax2.tick_params(axis='y', labelcolor="orange", labelsize=self.axes_fontsize)
            ax1.tick_params(axis='x', labelsize=self.axes_fontsize)

            ax2.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left', fontsize=self.legend_fontsize)


            if save:
                if save_name != None:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTime_"+self.model_name+"_"
                    file_name += save_name + ".png"
                    plt.savefig(file_name, bbox_inches="tight")
                else:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTime_"+self.model_name+".png"
                    plt.savefig(file_name, bbox_inches="tight")

        else:
            plt.xlabel("Time After Start of Sombrero [yrs]", fontsize=self.label_fontsize)
            plt.ylabel("Sombrero Width [km]", fontsize=self.label_fontsize)
            plt.xticks(fontsize=self.tick_fontsize)
            plt.yticks(fontsize=self.tick_fontsize)
            plt.grid()
            plt.scatter(time_steps-time_steps[0], np.asarray(width_list)/1e3)
            if save:
                if save_name != None:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTime_"+self.model_name+"_"
                    file_name += save_name + ".png"
                    plt.savefig(file_name, bbox_inches="tight")
                else:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTime_"+self.model_name+".png"
                    plt.savefig(file_name, bbox_inches="tight")
        plt.show()


    # A function to plot sombrero width on the major and minor axes as a function of time through sombrero
    def somb_width_time_ellipsoidal(self, time_steps_x, time_steps_y, save=False, save_name=None):

        #collect widths at each time step
        x_width_list = []
        y_width_list =[]

        for t in time_steps_x:
            model_time = self.get_timesteps(t)
            
            # get the width in the x direction
            (x0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time) #assign an x0 here before we do the rotation
            width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x0), np.asarray(vel_z)*self.ms_to_mmyr)
            x_width_list.append(width)

        for t in time_steps_y:
            model_time = self.get_timesteps(t)

            # get the width in the y direction 
            (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=np.pi/2.)
            width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x0), np.asarray(vel_z)*self.ms_to_mmyr)
            y_width_list.append(width)

        # make the plot
        plt.xlabel("Time After Start of Sombrero [yrs]", fontsize=self.label_fontsize)
        plt.ylabel("Sombrero Width [km]", fontsize=self.label_fontsize)
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        plt.grid()
        plt.scatter(time_steps_x-time_steps_x[0], np.asarray(x_width_list)/1e3, label="Minor Axis")
        plt.scatter(time_steps_y-time_steps_y[0], np.asarray(y_width_list)/1e3, label="Major Axis")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=self.tick_fontsize)

        if save:
            if save_name != None:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTimeEllipsoid_"+self.model_name+"_"
                file_name += save_name + ".png"
                plt.savefig(file_name, bbox_inches="tight")
            else:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/WidthTimeEllipsoid_"+self.model_name+".png"
                plt.savefig(file_name, bbox_inches="tight")
        plt.show()






#--------------Helper Plotting Functions-----------------------------------------#
            

# compare velocity profiles of different models 
def compare_widths(model_list, time_step_list, angle_avg=True, theta=None, color_list=None, key_list=None, title=None, norm_r=False, radial=False, norm_vel=None, width_markers=False, save=False, save_name=None):
    
    #list of angles to take profiles with (excluding 0 rad)
    angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

    #get profiles
    x0_list = []
    vz_list = []
    vr_list = []
    vtheta_list = []

    #get x_0
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0].get_data(1, theta=None)

    for i in range(len(model_list)):
        model = model_list[i]
        model_time = model.get_timesteps(time_step_list[i])

        vel_z_list = []
        vel_r_list = []
        vel_theta_list = []
        
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        vel_z_list.append(vel_z)
        vel_r_list.append(vel_r)
        vel_theta_list.append(vel_theta)
        
        if angle_avg:
            for theta in angles:
                (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)
                vel_z_list.append(vel_z)
                
                vel_r_list.append(vel_r)
                vel_theta_list.append(vel_theta)

        
        vel_z_avg = np.mean(np.asarray(vel_z_list), axis=0)
        vel_r_avg = np.mean(np.asarray(vel_r_list), axis=0)
        vel_theta_avg = np.mean(np.asarray(vel_theta_list), axis=0)
        x0_list.append(x_0)
        vz_list.append(vel_z_avg)
        vr_list.append(vel_r_avg)
        vtheta_list.append(vel_theta_avg)

    #begin to set up plot
    if norm_r:
        plt.xlabel("$r/r_{Source}$", fontsize=model_list[0].label_fontsize)
    else:
        plt.xlabel("Distance from Center [km]", fontsize=model_list[0].label_fontsize)
    
    if norm_vel != None:
        if radial:
            plt.ylabel("Normalized Radial Velocity", fontsize=model_list[0].label_fontsize)
            
        else:    
            plt.ylabel("Normalized Vertical Velocity", fontsize=model_list[0].label_fontsize)
    else:
        if radial:
            plt.ylabel("Radial Velocity [mm/yr]", fontsize=model_list[0].label_fontsize) 
        else:
            plt.ylabel("Vertical Velocity [mm/yr]", fontsize=model_list[0].label_fontsize)
   
    plt.xticks(fontsize=model_list[0].axes_fontsize)
    plt.yticks(fontsize=model_list[0].axes_fontsize)
    if title != None:
        plt.title(title, fontsize=model_list[0].title_fontsize)
    plt.grid()

    for i in range(len(model_list)):
        x_0 = x0_list[i]
        if radial:
            vel_plot = vr_list[i][x_0>0]
        else:
            vel_plot = vz_list[i][x_0>0]
        
        if norm_vel != None:
            if radial:
                norm = max(vr_list[norm_vel])*model_list[0].ms_to_mmyr
            else:
                norm = max(vz_list[norm_vel])*model_list[0].ms_to_mmyr
        else:
            norm = 1
        if key_list !=None:
            label = key_list[i]
        else:
            label = model_list[i].model_name
            
        #reverse and copy array
        vel_plot_rev = np.flip(vel_plot)
        vel_plot_full = np.concatenate((vel_plot_rev, vel_plot))
        
        x_0_rev = -1*np.flip(x_0[x_0>0])
        x_0_full = np.concatenate((x_0_rev, x_0[x_0>0]))

        
        if color_list != None:
            if norm_r:
                plt.plot(np.asarray(x_0_full)/25e3, (np.asarray(vel_plot_full)*model_list[0].ms_to_mmyr)/norm, label=label, linewidth=model_list[0].lw, color=color_list[i])
            else:
                plt.plot(np.asarray(x_0_full)/1e3, (np.asarray(vel_plot_full)*model_list[0].ms_to_mmyr)/norm, label=label, linewidth=model_list[0].lw, color=color_list[i])
        else:
            if norm_r:
                plt.plot(np.asarray(x_0_full)/25e3, (np.asarray(vel_plot_full)*model_list[0].ms_to_mmyr)/norm, label=label, linewidth=model_list[0].lw)
            else:
                plt.plot(np.asarray(x_0_full)/1e3, (np.asarray(vel_plot_full)*model_list[0].ms_to_mmyr)/norm, label=label, linewidth=model_list[0].lw)

        #get widths of profiles
        width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x_0_full), np.asarray(vel_plot_full)*model_list[0].ms_to_mmyr)
        print(label+": "+str(width/1e3)+" km")
        if width_markers:
            plt.scatter(pos_min_x/1e3, pos_min_y/norm, c="black", marker="|", s=50, zorder=20)
            plt.scatter(neg_min_x/1e3, neg_min_y/norm, c="black", marker="|", s=50, zorder=21)
    

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=model_list[0].legend_fontsize)

    if save:
        if save_name != None:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/CompareWidths_"
            if radial:
                file_name += "r_"
            file_name += save_name + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        else:
            if radial:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/CompareWidths_r.png"
            else:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/CompareWidths.png"
            plt.savefig(file_name, bbox_inches="tight")
    plt.show()

# function to plot both width and sombrero duration as a function of source radius
def plot_width_som_dur(model_list, time_step_list, save=False, save_name=None):

    #list of angles to take profiles with (excluding 0 rad)
    angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

    radius_list = []
    width_list = []
    som_dur_list = []

    #get profile widths and sombrero durations
    for i in range(len(model_list)):
        model = model_list[i]
        model_time = model.get_timesteps(time_step_list[i])

        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time)

        vz_temp = []
        vz_temp.append(vel_z)
        for theta in angles:
            (x_theta, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)
            vz_temp.append(vel_z)
        
        vz_avg = np.mean(np.asarray(vz_temp), axis=0)

        #get widths of profiles
        width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x), np.asarray(vz_avg)*model_list[0].ms_to_mmyr)
        width_list.append(width)

        #get sombrero duration
        som_dur_list.append(model.model.read_t_som()[len(model.model.read_t_som())-1])

        #get source radius
        radius_list.append(model.r_x) #keep with r_x for now, need to modify for ellipsoidal plot.

    #Make plot with two y axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.scatter(radius_list, np.asarray(width_list)/1e3, s=model_list[0].ms, c="blue")
    ax2.scatter(radius_list, som_dur_list, s=model_list[0].ms-40, c="orange")

    ax1.set_xlabel(r"$r_{source}$ [km]", fontsize=model_list[0].label_fontsize)
    ax1.set_ylabel("Sombrero Width [km]", fontsize=model_list[0].label_fontsize, color="blue")
    ax2.set_ylabel(r"$\Delta \tau_{som}$ [yrs]", fontsize=model_list[0].label_fontsize, color="orange")

    ax1.tick_params(axis='y', labelcolor="blue", labelsize=model_list[0].axes_fontsize)
    ax2.tick_params(axis='y', labelcolor="orange", labelsize=model_list[0].axes_fontsize)
    ax1.tick_params(axis='x', labelsize=model_list[0].axes_fontsize)

    if save:
        if save_name != None:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/som_dur_width_"
            file_name += save_name + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        else:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/somb_dur_width.png"
            plt.savefig(file_name, bbox_inches="tight")

    plt.show()

# function to plot both width and sombrero duration for both ellipsoidal axes as a function of the long ellipsoidal axis
def plot_width_som_dur_ellipsoidal(model_list, x_time_step_list, y_time_step_list, save=False, save_name=None):

    radius_list = []
    x_width_list = []
    x_som_dur_list = []
    y_width_list = []
    y_som_dur_list = []

    # get an x0 for the profiles
    (x0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0].get_data(1)

    # loop through all models
    for i in range(len(model_list)):
        model = model_list[i]
        x_model_time = model.get_timesteps(x_time_step_list[i])
        y_model_time = model.get_timesteps(y_time_step_list[i])


        # get x and y axis widths
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(x_model_time)
        x_width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x0), np.asarray(vel_z)*model_list[0].ms_to_mmyr)
        x_width_list.append(x_width)

        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(y_model_time, theta=np.pi/2.)
        y_width, pos_min_x, pos_min_y, neg_min_x, neg_min_y = find_sombrero_minima(np.asarray(x0), np.asarray(vel_z)*model_list[0].ms_to_mmyr)
        y_width_list.append(y_width)

        # get x and y axis sombrero durations
        x_som_dur_list.append(model.model.read_t_som()[len(model.model.read_t_som())-1]) # currently only x axis sombrero durations are on the spreadsheet

        som_time_steps = np.arange(model.model.spinup_time, model.model.spinup_time+model.model.cycles*model.model.T, 1)
        y_som_dur = model.get_som_dur(som_time_steps, use_ry=True, theta=np.pi/2., verbose=False, plot=False)
        y_som_dur_list.append(y_som_dur[len(y_som_dur)-1])

        # get y axis radius
        radius_list.append(model.r_y)
    
    # Make plot with two y axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.scatter(radius_list, np.asarray(x_width_list)/1e3, s=model_list[0].ms, c="blue", label="Minor Axis")
    ax1.scatter(radius_list, np.asarray(y_width_list)/1e3, s=model_list[0].ms, c="blue", marker='s', label="Major Axis")
    ax2.scatter(radius_list, x_som_dur_list, s=model_list[0].ms-40, c="orange", label="Minor Axis")
    ax2.scatter(radius_list, y_som_dur_list, s=model_list[0].ms-40, c="orange", marker='s', label="Major Axis")


    ax1.set_xlabel(r"$r_y$ [km]", fontsize=model_list[0].label_fontsize)
    ax1.set_ylabel("Sombrero Width [km]", fontsize=model_list[0].label_fontsize, color="blue")
    ax2.set_ylabel(r"$\Delta \tau_{som}$ [yrs]", fontsize=model_list[0].label_fontsize, color="orange")

    ax1.tick_params(axis='y', labelcolor="blue", labelsize=model_list[0].axes_fontsize)
    ax2.tick_params(axis='y', labelcolor="orange", labelsize=model_list[0].axes_fontsize)
    ax1.tick_params(axis='x', labelsize=model_list[0].axes_fontsize)

    ax1.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left', fontsize=model_list[0].legend_fontsize)
    ax2.legend(bbox_to_anchor=(1.15, 0.8), loc='upper left', fontsize=model_list[0].legend_fontsize)


    if save:
        if save_name != None:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/som_dur_width_ellipsoidal_"
            file_name += save_name + ".png"
            plt.savefig(file_name, bbox_inches="tight")
        else:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/somb_dur_width_ellipsoidal.png"
            plt.savefig(file_name, bbox_inches="tight")

    plt.show()


class ReadGPS:

    def __init__(self, path):

        self.path = path
        self.read_data()

    # read the GPS data file and store
    # data as class variables

    def read_data(self):

        df = pd.read_csv(self.path, delim_whitespace=True)

        # get name of station
        self.name = df['site'][0]

        # get lat and long of station from first entries in 
        # data set
        self.lat = df['_latitude(deg)'][0]
        self.long = df['_longitude(deg)'][0]

        # get data east, north and up and time
        self.dec_years = df['yyyy.yyyy']
        self.easting = df['__east(m)']
        self.northing = df['_north(m)']
        self.up = df['____up(m)']


#helper function given a sombrero profile, find the locations of the two minima and the quantitative width
def find_sombrero_minima(prof_x, prof_y):
    
    #get min on x>0 side
    prof_y_pos = list(prof_y[prof_x > 0])
    pos_idx = prof_y_pos.index(min(prof_y_pos)) + len(prof_y[prof_x<=0])-1 #adjust for shift in indices when only using positive y values
    pos_min_x = prof_x[pos_idx]
    pos_min_y = prof_y[pos_idx]

    #get min on x<0 side
    prof_y_neg = list(prof_y[prof_x<0])
    neg_idx = prof_y_neg.index(min(prof_y_neg)) #no need for adjustment here
    neg_min_x = prof_x[neg_idx]
    neg_min_y = prof_y[neg_idx]

    #get width
    width = pos_min_x - neg_min_x

    return width, pos_min_x, pos_min_y, neg_min_x, neg_min_y #all in m

# helper function to find the nearest point in a list to a given point
def get_nearest_point(x, y, x_list, y_list, no_same_point=False):
    
    best_dist = np.inf
    best_idx = -1

    for i in range(len(x_list)):
        curr_dist = np.sqrt((x-x_list[i])**2+(y-y_list[i])**2)
        if curr_dist == 0 and no_same_point:
            continue
        if curr_dist <= best_dist:
            best_dist = curr_dist
            best_idx = i
    return x_list[best_idx], y_list[best_idx]

# helper function to generate x-y locations of the surface projection of the CR
# assumed to be a cyllindrical CR unless CR_y is passed in then it's an ellipsoidal CR
def get_CR_points(CR_x, CR_y=None):

    theta_list = np.arange(0, 2*np.pi+0.1, 0.1)
    if CR_y == None:
        x = CR_x*np.cos(theta_list)
        y = CR_x*np.sin(theta_list)
    else:
        x = CR_x*np.cos(theta_list)
        y = CR_y*np.sin(theta_list)
    return x, y


#-------------------------Model comparison metrics and functions-------------------------------------#
def max_uplift_metric(model, time_steps):

    curr_max_vel = -np.inf

    for t in time_steps:
        model_time = model.get_timesteps(t)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time)

        if vel_z[x==0] > curr_max_vel: # max velocity should be in the center of the model
            curr_max_vel = vel_z[x==0] 

    return curr_max_vel[0]*model.ms_to_mmyr

def max_uplift_avg_metric(model, time_steps, debug=False):

    cur_max_vel = -np.inf
    cur_time = 0

    for t in time_steps:

        vel = model.get_average_vel_area(t, (model.r_x, model.r_y)) # get velocity averaged over the source area
        if vel > cur_max_vel:
            cur_max_vel = vel
            cur_time = t
        
    if debug:
        return (cur_max_vel, cur_time)
    return cur_max_vel



def max_sub_metric(model, time_steps):

    cur_max_sub = 0

    for t in time_steps:
        model_time = model.get_timesteps(t)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_0, vel_r, vel_theta) = model.get_data(model_time) #look at subsidence on minor and major axes
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_90, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2) 

        if vel_z_0[x==0] > 0:
            if min(min(vel_z_0), min(vel_z_90)) < cur_max_sub:
                cur_max_sub = min(min(vel_z_0), min(vel_z_90))
    
    return cur_max_sub*model.ms_to_mmyr

def max_sub_avg_metric(model, time_steps, debug=False, dr=0.75, CR_multiplyer=2):

    cur_max_sub = 0
    cur_time = 0
    cur_sub_dim = (model.r_x, model.r_y)

    for t in time_steps:
        
        center_vel = model.get_average_vel_area(t, (model.r_x, model.r_y)) # get velocity averaged over the source area

        # iterate through ellipse rings to find max subsidence
        r_i = model.r_x
        r_j = model.r_y
        while r_i <= model.r_x*CR_multiplyer and r_j <= model.r_y*CR_multiplyer:
            sub_vel = model.get_average_vel_area(t, (r_i+dr, r_j+dr), inner_axes=(r_i, r_j))

            if center_vel > 0 and sub_vel < cur_max_sub:
                cur_max_sub = sub_vel
                cur_time = t
                cur_sub_dim = (r_i, r_j)
            r_i += dr
            r_j += dr
    
    if debug:
        return (cur_max_sub, cur_time, cur_sub_dim)
    return cur_max_sub


# Returns the uplift/subsidence ratio at max uplift during sombrero
def max_ratio_metric(model, time_steps, CR_bounds=(3*13, 3*27.5), debug=False):

    ratio = np.inf
    ratio_time = 0
    ratio_pos = (0,0)
    cur_max_uplift = 0

    for t in time_steps:
        model_time = model.get_timesteps(t)
        (x0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_0, vel_r, vel_theta) = model.get_data(model_time) #look at subsidence on minor and major axes
        (x90, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z_90, vel_r, vel_theta) = model.get_data(model_time, theta=np.pi/2) 

        center_vel = model.get_average_vel_area(t, (model.r_x, model.r_y))

        if center_vel > 0:
            if center_vel > cur_max_uplift:
                cur_max_uplift = center_vel
                if min(min(vel_z_0), min(vel_z_90)) < 0:
                    sub = min(min(vel_z_0), min(vel_z_90))*model.ms_to_mmyr # get min subsidence during max uplift
                    ratio = np.abs(center_vel/sub)
                    if debug:
                        ratio_time = t
                        if min(vel_z_0) < min(vel_z_90):
                            pos_idx = np.argmin(vel_z_0)
                            theta = 0
                            # print(theta, x0[pos_idx]/1e3, vel_z_0[pos_idx]*model.ms_to_mmyr, sub)
                            ratio_pos = (round(x0[pos_idx]*np.cos(theta)/1e3, 2), round(x0[pos_idx]*np.sin(theta)/1e3, 2))
                        else:
                            pos_idx = np.argmin(vel_z_90)
                            theta = np.pi/2
                            # print(theta, x90[pos_idx]/1e3, vel_z_90[pos_idx]*model.ms_to_mmyr, sub)
                            ratio_pos = (round(x90[pos_idx]*np.cos(theta)/1e3, 2), round(x90[pos_idx]*np.sin(theta)/1e3, 2))
    
    if debug:
        return ratio, ratio_pos, ratio_time
    return ratio

def max_ratio_avg_metric(model, time_steps, debug=False, dr=0.75, CR_multiplier=2):

    cur_max_uplift = 0
    ratio = 0
    cur_time = 0
    

    for t in time_steps:
        
        center_vel = model.get_average_vel_area(t, (model.r_x, model.r_y)) # get velocity averaged over the source area

        # iterate through ellipse rings to find max subsidence
        r_i = model.r_x
        r_j = model.r_y
        itr_sub_dim = (r_i, r_j)
        cur_ratio_dim = itr_sub_dim
        cur_max_sub = 0
        while r_i <= model.r_x*CR_multiplier and r_j <= model.r_y*CR_multiplier:
            sub_vel = model.get_average_vel_area(t, (r_i+dr, r_j+dr), inner_axes=(r_i, r_j))
            print(r_i, r_j, sub_vel)

            if sub_vel < cur_max_sub:
                # print(sub_vel)
                cur_max_sub = sub_vel
                itr_sub_dim = (r_i, r_j)
            r_i += dr
            r_j += dr
        print(t, cur_max_sub, itr_sub_dim)
        if center_vel > cur_max_uplift and cur_max_sub < 0:
            cur_max_uplift = center_vel
            ratio = np.abs(center_vel/cur_max_sub)
            cur_time = t
            cur_ratio_dim = itr_sub_dim

    
    if debug:
        return (ratio, cur_time, cur_ratio_dim)
    return ratio

# Returns profile width (distance to 95% reduction in uplift at time of max uplift) along minor axis
def profile_width_metric(model, time_steps):

    curr_max_vel = -1
    for t in time_steps:
        model_time = model.get_timesteps(t)
        (x, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time)

        if vel_z[x==0] > curr_max_vel: # max velocity should be in the center of the model
            curr_max_vel = vel_z[x==0]
            max_vel_x = x[x >= 0] # get positive side of the profile
            max_vel_profile = vel_z[x >= 0] 

    width_idx = np.argmin(np.abs(max_vel_profile-0.05*curr_max_vel))
    return max_vel_x[width_idx]/1e3

# Returns relative volume change over given time interval
def rel_volume_change_metric(model, start_time, end_time):

    start_volume = model.get_ellipsoid_volume(start_time)
    end_volume = model.get_ellipsoid_volume(end_time)

    return (end_volume-start_volume)/start_volume

    


def pressure_rate_source_heatmap(model_list, metric, metric_name, CR_dim=(13, 27.5), save=True, bounds=None, color_map='seismic', use_dpdt=False, time_bounds=None):

    # get plotting variables from models
    pressure_rate_list = []
    geometry_list = []
    metric_list = []

    for model in model_list:
        if use_dpdt == False:
            pressure_rate_list.append(model.model.Delta_P/(0.75*model.model.T))
        else:
            pressure_rate_list.append(model.model.Delta_P)

        geometry_list.append(model.r_x/CR_dim[0])

        time_steps = np.arange(model.model.spinup_time, model.model.spinup_time+model.model.cycles*model.model.T+1, 1)
        if time_bounds != None:
            metric_list.append(metric(model, time_bounds[0], time_bounds[1]))
        else:
            metric_list.append(metric(model, time_steps))

    # make plot
    # plt.xlabel(r"$R_s/R_{CR}$", fontsize = model_list[0].axes_fontsize)
    # plt.ylabel(r"$\Delta P/(0.75 T)$ [kPa/yr]", fontsize = model_list[0].axes_fontsize)
    # plt.xticks(fontsize=model_list[0].tick_fontsize)
    # plt.yticks(fontsize=model_list[0].tick_fontsize)
    # plt.grid()

    # # plt.scatter(geometry_list, np.asarray(pressure_rate_list)/1e3, c=metric_list, s=100, cmap="jet")
    # if bounds != None:
    #     plt.scatter(geometry_list, np.asarray(pressure_rate_list)/1e3, c=metric_list, s=200, norm=clr.Normalize(vmin=bounds[0], vmax=bounds[1]), cmap="gist_rainbow")
    # else:
    #     plt.scatter(geometry_list, np.asarray(pressure_rate_list)/1e3, c=metric_list, s=200, cmap="gist_rainbow")

    # plt.colorbar().set_label(label=metric_name,size=model_list[0].label_fontsize)
        
    # make plot with seaborn

    # convert data to pandas dataframe
    pressure_rate_arr = np.round(np.asarray(pressure_rate_list)/(1e3), 2)
    geometry_arr = np.asarray(geometry_list)
    metric_arr = np.asarray(metric_list)

    df = pd.DataFrame.from_dict(np.array([geometry_arr,pressure_rate_arr,metric_arr]).T)
    df.columns = [r"$R_s/R_{CR}$",r"$dP/dt$ [kPa/yr]",metric_name]
    df[metric_name] = pd.to_numeric(df[metric_name])
    pivotted= df.pivot(r"$dP/dt$ [kPa/yr]",r"$R_s/R_{CR}$",metric_name)

    sns.set_theme(font_scale=2.4)
    if bounds != None:
        ax = sns.heatmap(pivotted,cmap=color_map, vmin=bounds[0], vmax=bounds[1], cbar_kws={'label': metric_name}, linewidths=0.5, annot=True)
    else:
        ax = sns.heatmap(pivotted,cmap=color_map, cbar_kws={'label': metric_name}, linewidths=0.5, annot=True)
    
    ax.invert_yaxis()

    if save:
        file_name = "/home/grantblock/Research/SMBPylith/Figures/pressure_rate_source_hm_"+metric_name.replace(' ', '_').replace('/', '_')+".png"
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def pressure_rate_viscosity_heatmap(model_list, metric, metric_name, save=True, bounds=None, color_map='seismic', time_bounds=None):
    # get plotting variables from models
    pressure_rate_list = []
    viscosity_list = []
    metric_list = []

    for model in model_list:
        pressure_rate_list.append(model.model.Delta_P*(model.model.T/model.model.P0)) #nondim pressure rate
        viscosity_list.append(model.model.tr/model.model.T) #nondim viscosity

        time_steps = np.arange(model.model.spinup_time, model.model.spinup_time+model.model.cycles*model.model.T+1, 1)
        if time_bounds != None:
            metric_list.append(metric(model, np.arange(time_bounds[0], time_bounds[1], 1)))
        else:
            metric_list.append(metric(model, time_steps))


    pressure_rate_arr = np.asarray(pressure_rate_list)
    viscosity_arr = np.asarray(viscosity_list)
    metric_arr = np.asarray(metric_list)

    df = pd.DataFrame.from_dict(np.array([viscosity_arr,pressure_rate_arr,metric_arr]).T)
    df.columns = [r"$\eta_{CR}/(T\mu_{CR})$",r"$dP/dt(T/P_0)$",metric_name]
    df[metric_name] = pd.to_numeric(df[metric_name])
    pivotted= df.pivot(r"$dP/dt(T/P_0)$",r"$\eta_{CR}/(T\mu_{CR})$",metric_name)

    sns.set_theme(font_scale=2.4)
    if bounds != None:
        ax = sns.heatmap(pivotted,cmap=color_map, vmin=bounds[0], vmax=bounds[1], cbar_kws={'label': metric_name}, linewidths=0.5, annot=True)
    else:
        ax = sns.heatmap(pivotted,cmap=color_map, cbar_kws={'label': metric_name}, linewidths=0.5, annot=True)
    
    ax.invert_yaxis()

    if save:
        file_name = "/home/grantblock/Research/SMBPylith/Figures/pressure_rate_visc_hm_"+metric_name.replace(' ', '_').replace('/', '_')+".png"
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


# Make plot to compare source depth vs. max uplift for multiple source sizes
def depth_uplift_multiplot(models_list, depth_lists, time_steps, save=True):

    # set up plot
    plt.xlabel("Source Depth [km]", fontsize=models_list[0][0].axes_fontsize)
    plt.ylabel("Maximum Vertical Uplift [mm/yr]", fontsize=models_list[0][0].axes_fontsize)
    plt.grid()
    plt.xticks(fontsize=models_list[0][0].tick_fontsize)
    plt.yticks(fontsize=models_list[0][0].tick_fontsize)

    for i in range(len(models_list)):
        models = models_list[i]
        depth_list = depth_lists[i]
        max_uplift = []
        for model in models:
            max_uplift.append(max_uplift_metric(model, time_steps))
        
        plt.scatter(depth_list, max_uplift, s=models_list[0][0].ms, label="Source Dim "+str(round(100*models[0].r_x/13, 2))+"% of CR dim")
        plt.plot(depth_list, max_uplift, linewidth=models_list[0][0].lw)
    
    plt.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left', fontsize=models_list[0][0].legend_fontsize)

    if save:
        file_name = "/home/grantblock/Research/SMBPylith/Figures/depth_uplift_multiplot.png"
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()

# Make plot to compare source depth vs. profile width
def depth_width_multiplot(models_list, depth_lists, time_steps, save=True):

    # set up plot
    plt.xlabel("Source Depth [km]", fontsize=models_list[0][0].axes_fontsize)
    plt.ylabel("Profile Width [km]", fontsize=models_list[0][0].axes_fontsize)
    plt.grid()
    plt.xticks(fontsize=models_list[0][0].tick_fontsize)
    plt.yticks(fontsize=models_list[0][0].tick_fontsize)

    for i in range(len(models_list)):
        models = models_list[i]
        depth_list = depth_lists[i]
        width = []
        for model in models:
            width.append(profile_width_metric(model, time_steps))
        
        plt.scatter(depth_list, width, s=models_list[0][0].ms, label="Source Dim "+str(round(100*models[0].r_x/13, 2))+"% of CR dim")
        plt.plot(depth_list, width, linewidth=models_list[0][0].lw)
    
    plt.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left', fontsize=models_list[0][0].legend_fontsize)

    if save:
        file_name = "/home/grantblock/Research/SMBPylith/Figures/depth_width_multiplot.png"
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


# plot non-dimensional velocity profiles from different models at a given time
def plot_nondim_profiles(model_list, time, add_models_list=None, burgers_models_list=None, add_models_times=None, add_colors=None, add_symbols=None, add_labels=None,
                          add_linestyles=None, add_line_colors=None, burgers_colors=None, burgers_symbols=None, burgers_labels=None, burgers_linestyles=None, theta=None):

    # get x_0 for plotting
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0][0].get_data(0, theta=None)

    # set up line styles and colors for plotting
    dpdt_colors = ["blue", "orange", "green", "brown"]
    visc_linestyles = ["solid", "dotted", "dashed"]

    # set up max vel plot
    max_norm_vels = []

    # plot 1E17 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[0]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        #Tr needs to be shifted by a factor 4!
        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        #norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[0], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)


    # plot 5E16 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[1]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        # norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[1], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)

    # plot 1E16 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[2]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        # norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[2], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)

    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel(r"$x/d_{source}$", fontsize=20)
    # plt.ylabel(r"$V_z d_{source}/\nu$", fontsize=20)
    # plt.ylabel(r"$V_z \sqrt{\nu T}/\nu$", fontsize=20)
    plt.ylabel(r"$V_z\left(\frac{dP}{dt}\right)^{-1}P_0/\sqrt{\nu \tau}$", fontsize=20)
    # plt.ylabel(r"$V_z/\sqrt{\frac{dP}{dt}\tau/\rho}$", fontsize=20)
    plt.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_profiles.png", bbox_inches="tight")
    plt.show()

    # plot trend with relax time
    norm_relax_times = 4*np.asarray([(model_list[0][0].model.tr/model_list[0][0].model.T), (model_list[1][0].model.tr/model_list[1][0].model.T), 
                                   (model_list[2][0].model.tr/model_list[2][0].model.T)])

    # calculate dP/dt
    v_norm = max_norm_vels[2]
    tr = 4*model_list[2][0].model.tr*3.154e+7 # s
    v_GPS = 50/model_list[2][0].ms_to_mmyr # m/s
    kinematic_visc = (tr*40e9)/2500 # m^2/s

    dpdt = (model_list[2][0].model.P0*v_GPS)/(v_norm*np.sqrt(kinematic_visc*tr))*(3.154e+7/1e3) # kPa/yr
    print(dpdt)

    # plt.scatter(norm_relax_times, max_norm_vels, label=r"Upside down sawtooth time series, $d_s$=5 km", c='blue', s=75)
    plt.scatter(norm_relax_times, max_norm_vels, label=r"Upside down sawtooth time series, $d_s$=5 km", c='gray', s=300, marker="^", edgecolors='black', zorder=6)
    plt.plot(norm_relax_times, max_norm_vels, c='gray', lw=3)

    # if there are additional models to plot on the trend
    if add_models_list != None:
        list_itr = 0
        for models in add_models_list:
            add_max_norm_vels = []
            add_norm_relax_times = []
            for model in models:
                model_time = model.get_timesteps(add_models_times[list_itr])
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)
                dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
                kinematic_visc = dynamic_visc/2500 #m^2/s
                norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
                norm_vel = vel_z*norm_factor
                add_max_norm_vels.append(max(norm_vel))
                add_norm_relax_times.append(4*model.model.tr/model.model.T)

            plt.scatter(add_norm_relax_times, add_max_norm_vels, c=add_colors[list_itr], marker=add_symbols[list_itr], label=add_labels[list_itr], s=300, edgecolors='black', zorder=7)
            plt.plot(add_norm_relax_times, add_max_norm_vels, c=add_line_colors[list_itr], linestyle=add_linestyles[list_itr], lw=3)
            list_itr += 1

    if burgers_models_list != None:
        MU = 1.5625e10 # models' shear modulus since update of Vs (Pa)
        list_itr = 0
        for models in burgers_models_list:
            norm_vels = []
            norm_relax_times = []
            for model, visc in models:

                model_time = model.get_timesteps(520)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time) 

                # get non-dimensional velocity
                dynamic_viscosity = visc # Pa*s
                kinematic_viscosity = dynamic_viscosity/2500 # m^2/s
                tr = dynamic_viscosity/MU # s
                norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_viscosity*tr)
                norm_vel = vel_z*norm_factor
                norm_relax_time = tr/(model.model.T*3.154e+7)
    
                norm_vels.append(max(norm_vel))
                norm_relax_times.append(norm_relax_time)

            plt.scatter(norm_relax_times, norm_vels, c=burgers_colors[list_itr], marker=burgers_symbols[list_itr], label=burgers_labels[list_itr], s=300, edgecolors='black', zorder=8)
            plt.plot(norm_relax_times, norm_vels, c='black', linestyle=burgers_linestyles[list_itr], lw=3)
            list_itr += 1

    
    plt.grid()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(r"$V_{max}\left(\frac{\Delta P}{\Delta t}\right)_{max}^{-1}P_0/\sqrt{\nu \tau}$", fontsize=40)
    plt.xlabel(r"$\tau/T$", fontsize=30)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=35, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_profiles_trend.png", bbox_inches="tight")
    plt.show()

# plot viscosity vs. max velocity trends for Brugers models
def plot_burgers_velocity_trend(model_list, time, visc_list):

    MU = 1.5625e10 # models' shear modulus since update of Vs (Pa)

    # iterate through models
    list_itr = 0
    norm_vels = []
    norm_relax_times = []
    for model in model_list:

        # get model data
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time)

        # get non-dimensional velocity
        dynamic_viscosity = visc_list[list_itr] # Pa*s
        kinematic_viscosity = dynamic_viscosity/2500 # m^2/s
        tr = dynamic_viscosity/MU # s
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_viscosity*tr)
        norm_vel = vel_z*norm_factor

        norm_vels.append(max(norm_vel))
        norm_relax_times.append(tr/(model.model.T*3.154e+7))

        # calculate dP/dt needed for each model to get to needed max velocity
        v_norm = max(norm_vel)
        v_GPS = 52/model.ms_to_mmyr # m/s
        dpdt = (model.model.P0*v_GPS)/(v_norm*np.sqrt(kinematic_viscosity*tr))*(3.154e+7/1e3) # kPa/yr
        print(dpdt, dynamic_viscosity)

        list_itr += 1

    plt.scatter(norm_relax_times, norm_vels)
    plt.plot(norm_relax_times, norm_vels)

    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel(r"$\left(V_z\left(\frac{dP}{dt}\right)^{-1}P_0/\sqrt{\nu \tau}\right)_{max}$", fontsize=20)
    plt.xlabel(r"$\tau/T$", fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_vel_trend_burgers.png", bbox_inches="tight")
    plt.show()

        


# plot velocity ratios as a function of relaxation time for sawtooth models
def plot_nondim_ratios(model_list, times):

    # set up line styles and colors for plotting
    dpdt_colors = ["blue", "orange", "green", "brown"]
    visc_linestyles = ["solid", "dotted", "dashed"]

    # seperate viscosities
    high_visc_models = model_list[0] # 1E17 Pa*s
    mid_visc_models = model_list[1] # 5E16 Pa*s
    low_visc_models = model_list[2] # 1E16 Pa*s

    nondim_tr_list = [low_visc_models[0].model.tr/low_visc_models[0].model.T, mid_visc_models[0].model.tr/mid_visc_models[0].model.T, high_visc_models[0].model.tr/high_visc_models[0].model.T]

    # get ratios based on dP/dt (order the models are in in each list) and plot
    for i in range(len(high_visc_models)):
        plot_list = [max_ratio_metric(low_visc_models[i], times), max_ratio_metric(mid_visc_models[i], times), max_ratio_metric(high_visc_models[i], times)]
        nondim_dpdt = low_visc_models[i].model.Delta_P*(low_visc_models[i].model.T/low_visc_models[i].model.P0)
        plt.plot(nondim_tr_list, plot_list, color=dpdt_colors[i], linewidth=4, zorder=-1)
        plt.scatter(nondim_tr_list, plot_list, color=dpdt_colors[i], s=75, edgecolors='black', label=r"$(dP/dt)(T/P_0)$="+str(nondim_dpdt))


    # get velocity ratio lists
    # high_visc_ratio_list = []
    # for model in high_visc_models:
    #     high_visc_ratio_list.append(max_ratio_metric(model, times))
    # mid_visc_ratio_list = []
    # for model in mid_visc_models:
    #     mid_visc_ratio_list.append(max_ratio_metric(model, times))
    # low_visc_ratio_list = []
    # for model in low_visc_models:
    #     low_visc_ratio_list.append(max_ratio_metric(model, times))

    # # get dP/dt list
    # dpdt_list = []
    # for model in high_visc_models:
    #     dpdt_list.append((model.model.Delta_P)*model.model.T/model.model.P0)

    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"$\tau/T$", fontsize=20)
    plt.ylabel("Velocity Ratio", fontsize=20)
    

    # # plot
    # plt.plot(dpdt_list, high_visc_ratio_list, color="blue", linewidth=4, zorder=-1)
    # plt.scatter(dpdt_list, high_visc_ratio_list, color="blue", label=r"$\tau$=0.1 yrs", s=75, edgecolors='black')

    # plt.plot(dpdt_list, mid_visc_ratio_list, color="orange", linewidth=4, zorder=-1)
    # plt.scatter(dpdt_list, mid_visc_ratio_list, marker='s', color="orange", label=r"$\tau$=0.05 yrs", s=75, edgecolors='black')

    # plt.plot(dpdt_list, low_visc_ratio_list, color="green", linewidth=4, zorder=-1)
    # plt.scatter(dpdt_list, low_visc_ratio_list, marker='X', color="green", label=r"$\tau$=0.01 yrs", s=75, edgecolors='black')

    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_ratios.png", bbox_inches="tight")
    plt.show()


# takes a list of model lists which should be related and only vary by viscosity and a list of times to evaluate the cross correlations,
# as well as a list of source depths for each set of models. Plots the distance to a given cross correlation threshold normalized by source depth
# as a function of relaxation time normalized by period.
def plot_anticorrelation_trend(model_lists, times_list, depths_list, colors_list, symbols_list, linestyle_list, labels_list, cc_thresh=0.5):

    # set up plot
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel(r"$\left(r(CC\leq"+str(cc_thresh)+r")-r_{s,x}\right)/d_s$", fontsize=20)
    plt.xlabel(r"$\tau/T$", fontsize=20)

    # loop through models list
    list_itr = 0
    for models in model_lists:
        # plotting variable lists
        norm_dists = []
        norm_relax = []

        for model in models:
            # distance at which the threshold is reached
            dist = -1
            x_points = np.arange(-model.r_x, -50, -0.5)
            for x_point in x_points:
                cc = model.get_cc_center_point(times_list[list_itr], (x_point, 0))
                if cc <= cc_thresh:
                    dist = np.abs(x_point + model.r_x)
                    break
            
            if dist != -1:
                norm_dists.append(dist/depths_list[list_itr])
                norm_relax.append(model.model.tr/model.model.T)
            else:
                print("Model", model.model.model_name, "doesn't have a cross correlation on the minor axis below", cc_thresh)
        plt.scatter(norm_relax, norm_dists, color=colors_list[list_itr], marker=symbols_list[list_itr], label=labels_list[list_itr], s=75)
        plt.plot(norm_relax, norm_dists, color=colors_list[list_itr], linestyle=linestyle_list[list_itr], lw=3)
        list_itr += 1
    
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_CC_dist_trend.png", bbox_inches="tight")
    plt.show()


# takes a list of model lists which should be related and only vary by viscosity and a list of times to evaluate the cross correlations,
# as well as a list of source depths for each set of models. Plots the distance to a given ratio threshold normalized by source depth
# as a function of relaxation time normalized by period.
def plot_ratio_trend(model_lists, times_list, depths_list, colors_list, symbols_list, linestyle_list, labels_list, ratio_thresh=None):
    # set up plot
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if ratio_thresh == None:
        plt.ylabel(r"$\left(r(\frac{V_0}{V_r}> 0)-r_{s,x}\right)/d_s$", fontsize=20)
    else:
        plt.ylabel(r"$\left(r(\frac{V_0}{V_r}\leq"+str(ratio_thresh)+r")-r_{s,x}\right)/d_s$", fontsize=20)
    plt.xlabel(r"$\tau/T$", fontsize=20)

    # loop through models list
    list_itr = 0
    for models in model_lists:
        # plotting variable lists
        norm_dists = []
        norm_relax = []

        for model in models:
            # distance at which the threshold is reached
            dist = -1
            x_points = np.arange(-model.r_x, -50, -0.5)
            for x_point in x_points:
                ratio = model.get_ratio_center_point(times_list[list_itr], (x_point, 0))
                
                if ratio_thresh == None:
                    if ratio != None:
                        dist = np.abs(x_point + model.r_x)
                        break 
                elif ratio <= ratio_thresh:
                    dist = np.abs(x_point + model.r_x)
                    break 

            if dist != -1:
                # print(model.model.model_name, dist)
                norm_dists.append(dist/depths_list[list_itr])
                norm_relax.append(model.model.tr/model.model.T)
            else:
                print("Model", model.model.model_name, "doesn't have a ratio on the minor axis below", ratio)

        plt.scatter(norm_relax, norm_dists, color=colors_list[list_itr], marker=symbols_list[list_itr], label=labels_list[list_itr], s=75)
        plt.plot(norm_relax, norm_dists, color=colors_list[list_itr], linestyle=linestyle_list[list_itr], lw=3)
        list_itr += 1

    plt.xscale("log")
    # plt.yscale("log")
    plt.legend(fontsize=20, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_ratio_dist_trend.png", bbox_inches="tight")
    plt.show()

# Make plot with average velocity of model points, with two (for now) stations
def plot_point_stationlist_avg(model_list, times, points_inner, points_outer, names, linestyles):
        

    # get base x prof
    (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
        vel_x_prof, vel_y_prof, vel_z_prof, 
        vel_r_prof, vel_theta_prof) = model_list[0].get_data_no_shift(0, theta=0)
        
    # set up plot
    # fig, ax = plt.subplots(2, 1, figsize=(14.8, 6),sharex=True)
    plt.grid()
    plt.yticks(fontsize=30)
    plt.ylabel("Normalized Velocity", fontsize=30)
    plt.xlabel("Time (yrs)", fontsize=30)
    plt.xticks(fontsize=30)

    # loop through model lists
    list_itr = 0

    for model in model_list:
        # loop through inner and outer points and collect
        points_inner_list = []
        for point in points_inner:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = model.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = model.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])

            points_inner_list.append(np.asarray(vel_array))

        # loop through outer points
        points_outer_list = []   
        for point in points_outer:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            for t in times:
                model_time = model.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = model.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])

            points_outer_list.append(np.asarray(vel_array))

        # get means of points
        points_inner_mean = np.mean(np.asarray(points_inner_list), axis=0)
        points_outer_mean = np.mean(np.asarray(points_outer_list), axis=0)

        # plot
        if list_itr == 0:
            plt.plot(times, np.asarray(points_inner_mean)/max(points_inner_mean), linestyle=linestyles[list_itr], label=str(names[list_itr])+" inner mean", lw=8, color='royalblue')
            plt.plot(times, np.asarray(points_outer_mean)/max(points_outer_mean), linestyle=linestyles[list_itr], label=str(names[list_itr])+" outer mean", lw=8, color='firebrick')
        else:
            plt.plot(times, np.asarray(points_inner_mean)/max(points_inner_mean), linestyle=linestyles[list_itr], label=str(names[list_itr])+" inner mean", lw=8, color='lightsteelblue')
            plt.plot(times, np.asarray(points_outer_mean)/max(points_outer_mean), linestyle=linestyles[list_itr], label=str(names[list_itr])+" outer mean", lw=8, color='lightcoral')

        list_itr += 1


    plt.legend(fontsize=20)
    plt.savefig("../../Figures/compare_models_mean.png", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_GPS_avg(GPS_in_mean_file, GPS_out_mean_file):

    # set up plot
    plt.grid()
    plt.yticks(fontsize=30)
    plt.ylabel("Normalized Velocity", fontsize=30)
    plt.xlabel("Time (yrs)", fontsize=30)
    plt.xticks(fontsize=30)

     # read inner and outer GPS mean files
    df_in = pd.read_csv(GPS_in_mean_file)
    GPS_in_times = df_in['time[yrs]'].values
    GPS_in_vel = df_in['velocity[m/yr]'].values

    df_out = pd.read_csv(GPS_out_mean_file)
    GPS_out_times = df_out['time[yrs]'].values
    GPS_out_vel = df_out['velocity[m/yr]'].values

    plt.plot(GPS_in_times, GPS_in_vel/max(GPS_in_vel), color="royalblue", lw=8,label="GPS inner mean")
        
    plt.plot(GPS_out_times, GPS_out_vel/max(GPS_out_vel), color="firebrick", lw=8, label="GPS outer mean")
    
    plt.legend(fontsize=20)

    plt.savefig("../../Figures/compare_GPS_mean.png", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

# compare models used for mesh resolution tests
def mesh_resolution_analysis(model_list, avg_S, plot_time, ratio_list=None):

    ms_to_mmyr = 3.154e+10

    # get x_0 for plotting profiles
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0].get_data(0, theta=None)

    # set up first plot
    plt.xlabel("Distance From Center of Source (km)", fontsize=25)
    plt.ylabel("Velocity (mm/yr)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()

    # iterate through models and plot profiles
    itr = 0
    max_vel_list = []
    for model in model_list:
        model_time = model.get_timesteps(plot_time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)

        max_vel_list.append(max(vel_z)*ms_to_mmyr) # collect max vels for second plot

        if ratio_list == None:
            plt.plot(x_0/1e3, vel_z*ms_to_mmyr, lw=5, label=r"$\langle S\rangle$="+str(avg_S[itr])+" km, t="+str(plot_time)+" yrs")
        else:
            plt.plot(x_0/1e3, vel_z*ms_to_mmyr, lw=5, label=r"$\langle S\rangle$="+str(avg_S[itr])+r" km, $R_v=$" + str(ratio_list[itr]) + ", t="+str(plot_time)+" yrs")

        itr += 1
    
    plt.xlim([-60, 60])
    plt.legend(fontsize=25, bbox_to_anchor=(1.25, 1.1))
    plt.savefig("/home/grantblock/Research/Yellowstone/Figures/res_test_legend.png", bbox_inches="tight")
    plt.show()

    # set up second plot
    plt.xlabel(r"$\langle S\rangle$ (km)", fontsize=30)
    plt.ylabel(r"$V_z(x=0,y=0,t="+str(plot_time)+r"$) (mm/yr)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()

    plt.plot(avg_S, max_vel_list, lw=6)
    plt.scatter(avg_S, max_vel_list, s=200, edgecolors='black')
    plt.show()


# Plot the model data residuals points along the model profile
def compare_residuals(models, mean_times, shift_time=None, stations=None, station_locs=None, symbol_list=None, label_list=None, size_list=None, colors=None): 

    # get base profile
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = models[0].get_data(0, theta=None)

    # iterate through models
    max_val = -np.infty
    min_val = np.infty
    for model, symbol, label, size, color in zip(models, symbol_list, label_list, size_list, colors):

        # get mean model velocities at each point
        station_names = []
        residuals = []
        loc = 0
        for point, station in zip(station_locs, stations):
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0 and y < 0:
                r *= -1
            elif x < 0 and y >= 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)

            # get mean  model velocity
            model_vel_list = []
            for t in mean_times:
                model_time = model.get_timesteps(t)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)
                point_idx = (np.abs(x_0 - r)).argmin() #get point index
                model_vel_list.append(vel_z[point_idx]*model.ms_to_mmyr)
            mean_model_vel = np.mean(np.asarray(model_vel_list), axis=0)

            # get mean station velocity
            df = pd.read_csv(station)
            GPS_times = df['time[yrs]'].values
            GPS_vel = df['velocity[m/yr]'].values
            
            time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)

            # if the station has data in the time cut, append to all of the lists
            if len(GPS_vel[time_cut]) > 0:

                GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3
                name = station[0:-4]

                station_names.append(name)

                residual = 100*(GPS_mean_vel-mean_model_vel)/GPS_mean_vel

                if residual > max_val:
                    max_val = residual
                if residual < min_val:
                    min_val = residual

                residuals.append(residual)
                loc+=1

        plt.scatter(np.arange(loc), residuals, c=color, s=size, zorder=10, marker=symbol, label=label, edgecolors='black')


    # plot
    plt.grid()
    plt.ylabel("Residual (% diff)", fontsize=40)
    plt.xlabel("Station", fontsize=40)
    plt.ylim([min_val-10000, max_val+1000])
    plt.xticks(np.arange(loc), station_names, rotation ='horizontal', fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)
    plt.yscale('symlog')

    plt.show()

    


                