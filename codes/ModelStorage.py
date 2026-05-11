import pandas as pd
import numpy as np
import math


# Class to read in model parameters from csv and write derived model
# parameters to the csv
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.read_csv()

    #function to read csv and store all given parameters
    def read_csv(self):

        df = pd.read_csv("run_list.csv")

        #get row by model name
        model_row = df.loc[df['name'] == self.model_name]

        #store given params as class variables
        self.path = np.array(model_row['path'])[0]
        self.P0 = np.asarray(model_row['P0'])[0]
        self.Delta_P = np.asarray(model_row['Delta_P'])[0]
        self.T = np.asarray(model_row['T'])[0]
        self.tr = np.asarray(model_row['tr'])[0]
        self.run_time = np.asarray(model_row['run_time'])[0]
        self.spinup_time = np.asarray(model_row['spinup_time'])[0]
        self.output_dt = np.asarray(model_row['output_dt'])[0]
        self.cycles = np.asarray(model_row['cycles'])[0]

        if np.asarray(model_row['nested'])[0] == 1:
            self.nested = True
        else:
            self.nested = False
        if np.asarray(model_row['sawtooth'])[0] == 1:
            self.sawtooth = True
        else:
            self.sawtooth = False
        if np.asarray(model_row['inv_sawtooth'])[0] == 1:
            self.inv_sawtooth = True
        else:
            self.inv_sawtooth = False

        #extra formatting if output dt is a list of 2 numbers
        if type(self.output_dt) == str:
            if " " in self.output_dt:
                output_list_str = self.output_dt.split(" ")
                output_list = [float(output_list_str[0]), float(output_list_str[1])]
                self.output_dt = output_list
            else:
                self.output_dt = float(self.output_dt)

    #function to write derived parameters to the csv
    def update_csv(self, tc=None, t_som=None, som_mag=None, Delta_v=None):

        df = pd.read_csv("run_list.csv")


        #update derived parameter values
        if tc != None and type(np.asarray(df.loc[df['name'] == self.model_name]['tc'])[0]) != str:
            if math.isnan(np.asarray(df.loc[df['name'] == self.model_name]['tc'])[0]):
                df.loc[df['name'] == self.model_name, 'tc'] = tc
        
        if t_som != None and type(np.asarray(df.loc[df['name'] == self.model_name]['t_som'])[0]) != str:
            if math.isnan(np.asarray(df.loc[df['name'] == self.model_name]['t_som'])[0]):
                if type(t_som) == list:
                    t_som_str = ""
                    if len(t_som) == 0:
                        t_som_str+="0"
                    for t_som_elm in t_som:
                        t_som_str += str(t_som_elm)+" "
                    df.loc[df['name'] == self.model_name, 't_som'] = t_som_str
                else: 
                    df.loc[df['name'] == self.model_name, 't_som'] = t_som

        if som_mag != None and type(np.asarray(df.loc[df['name'] == self.model_name]['som_mag'])[0]) != str:
            if math.isnan(np.asarray(df.loc[df['name'] == self.model_name]['som_mag'])[0]):
                if type(som_mag) == list:
                    mag_str = ""
                    if len(som_mag) == 0:
                        mag_str+="0"
                    for mag in som_mag:
                        mag_str += str(mag)+" "
                    #df.loc[df['name'] == self.model_name, 'som_mag'] = str(som_mag[0])+" "+str(som_mag[1])
                    df.loc[df['name'] == self.model_name, 'som_mag'] = mag_str
                else: 
                    df.loc[df['name'] == self.model_name, 'som_mag'] = som_mag

        if Delta_v != None and type(np.asarray(df.loc[df['name'] == self.model_name]['Delta_v'])[0]) != str:
           if math.isnan(np.asarray(df.loc[df['name'] == self.model_name]['Delta_v'])[0]):
                df.loc[df['name'] == self.model_name, 'Delta_v'] = Delta_v

        df.to_csv("run_list.csv", index=False)


    # functions to get derived parameters after they've been written
    def read_tc(self):
        df = pd.read_csv("run_list.csv")

        #get row by model name
        model_row = df.loc[df['name'] == self.model_name]

        tc = np.array(model_row['tc'])[0]
        if math.isnan(tc):
            print("Warning: tc not written to csv, please run get_tc() first. Nothing will be returned.")
            return
        return tc
    
    def read_t_som(self):
        df = pd.read_csv("run_list.csv")

        #get row by model name
        model_row = df.loc[df['name'] == self.model_name]
        t_som = np.array(model_row['t_som'])[0]

        #if t_som is a string of two numbers parse and return a list
        if type(t_som) == str:
            t_som_list = []
            for t_som_str in t_som.split(" "):
                t_som_list.append(float(t_som_str))
            return t_som_list
        elif math.isnan(t_som):
            print("Warning: t_som not written to csv, please run get_t_som() first. Nothing will be returned.")
            return
        return t_som
    
    def read_som_mag(self):
        df = pd.read_csv("run_list.csv")

        #get row by model name
        model_row = df.loc[df['name'] == self.model_name]
        som_mag = np.array(model_row['som_mag'])[0]

        #if som_mag is a string of two numbers parse and return a list
        if type(som_mag) == str:
            som_mag_list = []
            for som_mag_str in som_mag.split(" "):
                som_mag_list.append(float(som_mag_str))
            return som_mag_list
        elif math.isnan(som_mag):
            print("Warning: som_mag not written to csv, please run get_som_mag() first. Nothing will be returned.")
            return
        return som_mag
    
    def read_delta_v(self):
        df = pd.read_csv("run_list.csv")

        #get row by model name
        model_row = df.loc[df['name'] == self.model_name]

        delta_v = np.array(model_row['Delta_v'])[0]
        if math.isnan(delta_v):
            print("Warning: delta_v not written to csv, please run get_delta_v() first. Nothing will be returned.")
            return
        return delta_v


    
# test = Model("Sawtooth_Pres18")
# print(test.read_t_som())

        


