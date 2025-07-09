  # -*- coding: utf-8 -*-
"""
@author: Justin France
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd



def load_data(data_set1, data_set2, data_set3):
    """ load in data from exel sheet """
    data_1 = np.loadtxt(data_set1, delimiter=",", skiprows=1)
    
    data_2 = np.loadtxt(data_set2, delimiter=",", skiprows=1)
   
    data_3 = np.loadtxt(data_set3, delimiter=",", skiprows=1)
    # print("data_set1")
    # print(data_1)
    # print('\n')
    
    # print("data_set2")
    # print(data_2)
    # print('\n')
    
    # print("data_set3")
    # print (data_3)
    # print("\n")
    return(data_1, data_2, data_3)



def process_data(data, cf_m, fist_pizo, second_pizo, cf_ms):
    """ slect data set for pizeometer  for the venturi, diffuser , orifice plate 
    and convert values to SI units and caculate the diffrence in head pressure"""
    pizo_data = np.array(data[:,[fist_pizo,second_pizo]])
    pizodata = pizo_data * cf_m
    head = np.array(pizodata[:,[0]])-np.array(pizodata[:,[1]]) 
    
    rotometer_flow_rate = np.array(data[:,[0]])
    
    
    elc_flow_data = (np.array(data[:,[7]])) * cf_ms
    elc_flow_rate = np.round(elc_flow_data, 6)
    
    
    
    # print(elc_flow_rate)
    #print(data) 
    # print("Head pizometer",str(fist_pizo),"-pizometer",str(second_pizo))
    # print(pizodata)
    # print("\n")
    # print(head)
    # print('\n')
    return rotometer_flow_rate, head, elc_flow_rate
    
     
def cal_pressure_diff(g, density, head):
    """ Caculates te change in pressure in [Pa g] from: density of water, gravity and the change in head"""
    pressure_diff = g * density * head
    return pressure_diff


def cal_reynolds_number(Q, D, v, A):
    """ caculate reynolds number: Re = uD/v to determine the flow characteristic."""
    U = Q / A
    #print(U)
    Re = np.round(((U*D)/v) ,2)
    return Re


def cal_cd_cofficient(Re_cal, Re_low_lim, Re_up_lim, c_d):
    """ compaires the caculated Reynalds number agenst the provided Reynalds
    numbers an assigns the approprate Cd cofficient"""
    # print("Re_low_lim", Re_low_lim)
    # print("Re_up_lim", Re_low_lim)
    # print("c_d", c_d, "\n")
    
    i = 0
    j = 0
    cd_cofficients = []
    for i in range(len(Re_cal)):
        for j in range(len(Re_low_lim)):
            if (Re_cal[i] > Re_low_lim[j]) and (Re_cal[i] <= Re_up_lim[j]):
                #print("Re", Re_cal[i], ">", Re_low_lim[j],  "&  Re", Re_cal[i], "<", Re_up_lim[j]  )
                cd_cofficients.append(c_d[j])
                break        
    return cd_cofficients
            
            
def cal_flow_rate(cd_vals, A_1, A_2, density, p_diff):
    """ caculates the flow rate from fuge factor, area profiles, density and change in pressure to caculate 
    the flow rate across the conponent  U = cd *A_2 *sqrt( (2 *p_diff) / (1- A_2**2 / A_1**2"""  
    # print(cd_vals)
    # print(A_1)  
    # print(A_2)
    # print(density)
    # print(p_diff)    
    for cd_val in cd_vals:
        flow_rate =  cd_val * A_2 *np.sqrt((2 * np.array(p_diff)) /(density * ( 1 - ((A_2**2) / (A_1**2)))))
    #print(flow_rate)
    return flow_rate


def uncertainties_pressure_diff(venturi_head, piezo_error,density_error, gravity, density):
     """ caculate the uncertanties for the diffrence in pressure"""
     pressure_error = np.sqrt( (gravity * venturi_head * density_error)**2 + (density * gravity * piezo_error)**2 + (density * gravity * piezo_error)**2 )
     return pressure_error
     

def Uncertanties_area(diameter, diameter_error):
    """ caculate unxcedrtanties for area"""
    area_error = (np.pi * diameter) / 2 * diameter_error
    return area_error

def Uncertanties_density(density_water, flow_rate):
        """ caculate the error of density"""
        density_uncertantity = (0.5 / density_water) * flow_rate
        return density_uncertantity
    
def Uncertanties_flow_area_1(A_1, A_2, flow_rate):
    """ caculate the area_1 uncertanty with respect to flow"""
    error_flow_area_1 = (1) / ( (1 / A_1**2 - 1 / A_2**2) * A_1**3) * flow_rate
    return error_flow_area_1 
    
    
def Uncertanties_flow_area_2(A_1, A_2, flow_rate):
     """ caculate the area_2 uncertanty with respect to flow"""
     error_flow_area_2 = (1) / ( (1 / A_2**2 - 1 / A_1**2) * A_2**3) * flow_rate
     # print(flow_rate)
     # print(error_flow_area_2)
     return error_flow_area_2 
   
def uncertanties__flow_calutation(flow_rate, pressure_diff, pressure_err, density_uncertantity, density_err, A_1_flow, A_2_flow, Area_1_err, Area_2_err):
    """caculate the error for the flow rate caculation"""
    error_flow_upper = np.sqrt( (((0.5 / pressure_diff) * flow_rate) * pressure_err)**2 + (density_uncertantity * density_err)**2 + ( A_1_flow * Area_1_err )**2 + (A_2_flow * Area_2_err )**2)
    error_flow_lower = np.sqrt( (((0.5 / pressure_diff) * flow_rate) * pressure_err)**2 + (density_uncertantity * density_err)**2 + ( A_1_flow * Area_1_err )**2 + (A_2_flow * Area_2_err )**2)
    return error_flow_upper, error_flow_lower


def error_rotameter_caculation(rotometer_flow_rate, rotameter_error):
    """ caculate the uncertanties for the rotameter readings"""
    rotameter_uncertanty = rotameter_error / rotometer_flow_rate
    return rotameter_uncertanty


''' plot_flow_rates(venturi_flow_rate, orifice_flow_rate, flow_rate, error_flow_venturi_upper, error_flow_orifice_upper)'''
def plot_flow_rates(venturi_flow_rate, orifice_flow_rate, flow_rate, error_flow_venturi, error_flow_orifice):
    """ plot the ventri and orifice flow rates agenst the electronic flow meter flow rate mesaruments
    with an additional 1 to 1 line)"""
    elc_flow_rate = np.array(flow_rate)
    
    axes = plt.axes()
    axes.scatter(elc_flow_rate, venturi_flow_rate, s=1, marker=".", color="blue", label="Venturi")
    axes.scatter(elc_flow_rate, orifice_flow_rate, s=1,  marker=".", color="red", label="Orifice plate")
    axes.plot(np.arange(0, 0.0006, 0.00005),np.arange(0, 0.0006, 0.00005), color="black", linewidth=0.75, label="1-1 line")
    axes.legend()
    
    # print("error_flow_venturi\n", error_flow_venturi)
    # print(error_flow_venturi[:, 0])
    
    # # print(venturi_flow_rate)
    
    axes.errorbar(elc_flow_rate, venturi_flow_rate[:, 0], yerr = (error_flow_venturi[:, 0]), fmt='.', color="blue", label="Venturi", capsize=5)
    axes.errorbar(elc_flow_rate,  orifice_flow_rate[:, 0], yerr = (error_flow_orifice[:, 0]), fmt='.', color="red", label="Orifice", capsize=5)
    
    x_labels = np.arange(0, 0.0006, 0.00005)
    x_labels_5_sf = [f"{val:.5f}" for val in x_labels]
    axes.set_xticks(np.arange(0, 0.0006, 0.00005))
    axes.set_xticklabels(x_labels_5_sf, rotation=(20), fontsize=10)
    axes.set_xlabel(r"Flow rate from electronic flow meter [m$^3$/s]", fontsize=10)
    
    axes.set_yticks(np.arange(0, 0.0008, 0.00005))
    y_labels = np.arange(0, 0.0008, 0.00005)
    y_labels_5_sf = [f"{val:.5f}" for val in y_labels]
    axes.set_yticklabels(y_labels_5_sf, fontsize=10)
    axes.set_ylabel(r"Flow rate from pressure drop [m$^3$/s]", fontsize=10)
    axes.grid()
    plt.show()
   
    
def plot_rotameter_elc_flow_rates(rotometer_flow_rate,flow_rate, cf_m3s, rotameter_uncertanty):
    """ plot electronic flow rates agenst the rotameter flow rates"""
    # print(rotometer_flow_rate)
    rotameter_SI = np.array(rotometer_flow_rate) * cf_m3s
    # print(rotameter_SI) 
    # print(flow_rate)
    flow_rate = np.ravel(flow_rate)
    rotameter = np.ravel(rotameter_SI)
    
    axes = plt.axes()
    axes.scatter(flow_rate, rotameter, marker=".", color="blue", label="flow meter Vs rotameter")
    
    axes.errorbar(flow_rate, rotameter_SI[:, 0], yerr = (rotameter_uncertanty[:, 0]), fmt='.', color="blue", capsize=5)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(flow_rate, rotameter)
    r_squared = r_value**2
    
   
    axes.axline(xy1=(0, intercept), slope=slope, color="red", linestyle="--", label=f"Linear regression: $R^2 = {r_squared:.3f}$")
    axes.legend()
    
    x_labels = np.arange(0, 0.0006, 0.00005)
    x_labels_5_sf = [f"{val:.5f}" for val in x_labels]
    axes.set_xticks(np.arange(0, 0.0006, 0.00005))
    axes.set_xticklabels(x_labels_5_sf, rotation=(20), fontsize=10)
    axes.set_xlabel(r"Flow rate from electronic flow meter [m$^3$/s]", fontsize=10)
    
    axes.set_yticks(np.arange(0, 0.0008, 0.00005))
    y_labels = np.arange(0, 0.0008, 0.00005)
    y_labels_5_sf = [f"{val:.5f}" for val in y_labels]
    axes.set_yticklabels(y_labels_5_sf, fontsize=10)
    axes.set_ylabel(r"Flow rate from rotameter [m$^3$/s]", fontsize=10)
    
    axes.grid()
    plt.show()
    
   
def format_processed_data(venturi_flow_rate,
                          error_flow_venturi,
                          pressure_diff_venturi, 
                          error_pressure_diff_venturi,
                          Re_venturi,orifice_flow_rate, 
                          error_flow_orifice,
                          pressure_diff_orifice, 
                          error_pressure_diff_orifice, 
                          Re_orifice, flow_rate):
    """create a dictionaryfor processed data for the venturi and orifice plate: 
        flow rate, flowrate uncertainty, pressure, pressure uncertainty,  Reynalds number"""
    #delta_P = "\u0394P"
    #plus_minus = "\u00B1"
    processed_data = {'Venturi Q [m\u00b3/s]':venturi_flow_rate[:, 0],
                      'Venturi Q error \u00B1 [m\u00b3/s]':error_flow_venturi[:, 0], 
                      'Orifice Q [m\u00b3/s]':orifice_flow_rate[:, 0],
                      'Orifice Q error \u00B1 [m\u00b3/s]':error_flow_orifice[:, 0],
                      'Electronic flow rate [m\u00b3/s]':flow_rate[:, 0],
                      'Venturi \u0394P [Pa]':pressure_diff_venturi[:, 0], 
                      'Error venturi \u00B1\u0394P [Pa]':error_pressure_diff_venturi[:, 0], 
                      'Orifice \u0394P [Pa]':pressure_diff_orifice[:, 0],
                      'Error orifice \u00B1\u0394P [Pa]':error_pressure_diff_orifice[:, 0],
                      'Reynalds number venturi':Re_venturi[:, 0],
                      'Reynalds number orifice':Re_orifice[:, 0]}
    
    #print(processed_data)
    return processed_data
        
def export_data_to_exel(data):
    """ export dictionary as to make an exel spread sheet"""
    df = pd.DataFrame(data)
    # Export the DataFrame to an Excel file
    df.to_excel("processed_data.xlsx", index=False)  # Save to file without including row indices

    print("Data exported successfully to 'processed_data.xlsx'")
    
    
def main():
    data_set1 = "LabA_data1.csv"
    data_set2 = "labA_data2.csv"
    data_set3 = "LabA_data3.csv"
    
    cf_M = 0.001 # conversion factor mm to M
    cf_ms = 1.6666667*10**-5 # conversion factor from liters per minute to meters per second
    cf_m3s = 1*10**-6 # conversion factor from mm/min to m^3 / s
    
    gravity = 9.81  # gravity in m/s^2
    kinematic_v_h20 = 1.007*10**-6 # kinematic velocity of water at 20 degrees celcius 
    
    d_1_venturi = 0.026     # diameter in meters for the pipe upstream of venturi throat
    d_2_venturi = 0.016     # diameter in meters for the venturi throat
    A_1_venturi = (np.pi * d_1_venturi**2) / 4
    A_2_venturi = (np.pi * d_2_venturi**2) / 4
    
    d_1_orifice = 0.0519 #diameter in meters for the pipe down stream of orifice
    d_2_orifice =0.02 #diameter in meters of the orifice plate hole
    A_1_orifice = (np.pi * d_1_orifice ** 2) / 4
    A_2_orifice = (np.pi * d_2_orifice ** 2) / 4
    
    
    Re_low_lim = [1, 75000, 150000, 250000, 400000, 1000000, 2000000]  # lower threshold values of reynalds number
    Re_up_lim = [75000, 150000, 250000, 400000, 1000000, 2000000, 100000000] # upper threshold values of reynalds number
    c_d  = [0.97, 0.977, 0.992, 0.998, 0.995, 1.00, 1.01]   #fuge factor cofficient which corospond to the a specifice range of reynald numbers
    
    c_d_orifice = np.full(9, 0.61) # from (D_2 / D_1)  and ( renalds number < 10^4 ) determines a c_d orifice value of 0.61 from 314lab manual
    
    
    pizo_1 = 1  # venturi pizometer
    pizo_2 = 2  # venturi pizometer
    pizo_3 = 3  # diffuser pizometer
    pizo_4 = 4  # diffuser pizometer
    pizo_5 = 5  # orifice plate pizometer
    pizo_6 = 6  # orifice plate pizometer
    
    
    piezo_error = 0.0005 # mesurment uncertainty for pizometers [m]
    density_error = 0.25  # change in temperature of 1 degree celcius changes density of water.
    diameter_error = 0.0005 # error in mesurment from mesuring diameters. 
    rotameter_error = 0.0005 # error in mesuring rotameter
    
    
    data_1, data_2, data_3 = load_data(data_set1, data_set2, data_set3)
    
    """--------------------------------------------------------------------------"""
    """ uncomment the following blocks of code to see the results of test1, test2, test3"""
    
    # rotometer_flow_rate, venturi_head, flow_rate =  process_data(data_1, cf_M, pizo_1, pizo_2, cf_ms)
    # rotometer_flow_rate, diffuser_head, flow_rate = process_data(data_1, cf_M, pizo_3, pizo_4, cf_ms)
    # rotometer_flow_rate, orifice_head, flow_rate = process_data(data_1, cf_M, pizo_5, pizo_6, cf_ms)
    # density_water = 998.2 # density of water at 20 degrees celcius [kg/m^3]
    
    
    # rotometer_flow_rate ,venturi_head, flow_rate =  process_data(data_2, cf_M, pizo_1, pizo_2, cf_ms)
    # rotometer_flow_rate, diffuser_head, flow_rate = process_data(data_2, cf_M, pizo_3, pizo_4, cf_ms)
    # rotometer_flow_rate, orifice_head, flow_rate = process_data(data_2, cf_M, pizo_5, pizo_6, cf_ms)
    # density_water = 998.45 # density of water at 21 degrees celcius [kg/m^3]
    
    
    rotometer_flow_rate, venturi_head, flow_rate =  process_data(data_3, cf_M, pizo_1, pizo_2, cf_ms)
    rotometer_flow_rate, diffuser_head, flow_rate = process_data(data_3, cf_M, pizo_3, pizo_4, cf_ms)
    rotometer_flow_rate, orifice_head, flow_rate = process_data(data_3, cf_M, pizo_5, pizo_6, cf_ms)
    density_water = 998.575 # density of water at 22 degrees celcius [kg/m^3]
    """-----------------------------------------------------------------------"""
    
    pressure_diff_venturi = cal_pressure_diff(gravity, density_water, venturi_head)
    pressure_diff_diffuser = cal_pressure_diff(gravity, density_water, diffuser_head)
    pressure_diff_orifice  = cal_pressure_diff(gravity, density_water, orifice_head)
    
    
    Re_venturi = cal_reynolds_number(flow_rate, d_2_venturi, kinematic_v_h20, A_2_venturi)
    Re_orifice = cal_reynolds_number(flow_rate, d_2_orifice, kinematic_v_h20, A_2_orifice)
    
    
    venturi_cd = cal_cd_cofficient(Re_venturi, Re_low_lim, Re_up_lim, c_d)
    orifice_cd = cal_cd_cofficient(Re_orifice, Re_low_lim, Re_up_lim, c_d)
    
    
    venturi_flow_rate = cal_flow_rate(venturi_cd, A_1_venturi, A_2_venturi, density_water, pressure_diff_venturi)
    orifice_flow_rate = cal_flow_rate( c_d_orifice , A_1_orifice, A_2_orifice , density_water, pressure_diff_orifice)
    
    
    density_uncertantity = Uncertanties_density(density_water, flow_rate)
    
    error_pressure_diff_venturi = uncertainties_pressure_diff(venturi_head, piezo_error, density_error, gravity, density_water)
    error_area_1_venturi = Uncertanties_area(d_1_venturi, diameter_error)
    error_area_2_venturi = Uncertanties_area(d_2_venturi, diameter_error)
    
    error_pressure_diff_orifice = uncertainties_pressure_diff(orifice_head, piezo_error, density_error, gravity, density_water)
    error_area_1_orifice = Uncertanties_area( d_1_orifice, diameter_error)
    error_area_2_orifice = Uncertanties_area(d_2_orifice , diameter_error)
    
    flow_area_1_error_venturi = Uncertanties_flow_area_1(A_1_venturi, A_2_venturi, flow_rate)
    flow_area_1_error_orifice = Uncertanties_flow_area_1(A_1_orifice, A_2_orifice, flow_rate)
    
    flow_area_2_error_venturi = Uncertanties_flow_area_2(A_1_venturi, A_2_venturi, flow_rate)
    flow_area_2_error_orifice = Uncertanties_flow_area_2(A_1_orifice, A_2_orifice, flow_rate)
    
    """uncertanties__flow_calutation(flow_rate, pressure_diff, pressure_err, density_uncertantity, density_err, A_1, A_2, Area_1_err, Area_2_err):"""
    error_flow_venturi_upper, error_flow_venturi_lower = uncertanties__flow_calutation(venturi_flow_rate, pressure_diff_venturi,
                                                                                       error_pressure_diff_venturi,
                                                                                       density_uncertantity, 
                                                                                       density_error, flow_area_1_error_venturi,  
                                                                                       flow_area_2_error_venturi,
                                                                                       error_area_1_venturi, 
                                                                                       error_area_2_venturi)
    error_flow_orifice_upper, error_flow_orifice_lower = uncertanties__flow_calutation(orifice_flow_rate,
                                                                                       pressure_diff_orifice,  
                                                                                       error_pressure_diff_orifice,
                                                                                       density_uncertantity, density_error,
                                                                                       flow_area_1_error_orifice, 
                                                                                       flow_area_2_error_orifice,
                                                                                       error_area_1_orifice,
                                                                                       error_area_2_orifice)
    
    rotameter_uncertanty = error_rotameter_caculation(rotometer_flow_rate, rotameter_error)
    
    plot_flow_rates(venturi_flow_rate, orifice_flow_rate, flow_rate, error_flow_venturi_upper, error_flow_orifice_upper, )
    plot_rotameter_elc_flow_rates(rotometer_flow_rate, flow_rate, cf_m3s, rotameter_uncertanty)
    
   
    data_to_export = format_processed_data(venturi_flow_rate, error_flow_venturi_upper,
                                           pressure_diff_venturi, 
                                           error_pressure_diff_venturi,
                                           Re_venturi,  
                                           orifice_flow_rate,
                                           error_flow_orifice_upper, 
                                           pressure_diff_orifice,
                                           error_pressure_diff_orifice,
                                           Re_orifice, flow_rate)
    
    export_data_to_exel(data_to_export)
    
    
    # density_t1 = cal_density_T(VTEC_water, density_water_std, std_temp, water_T1)
    # density_t2 = cal_density_T(VTEC_water, density_water_std, std_temp, water_T2)
    # density_t3 = cal_density_T(VTEC_water, density_water_std, std_temp, water_T3)
    
    #print(flow_rate)
    #print("Venturi head\n",venturi_head, "\n")
    #print(diffuser_head, "\n")
    #print(orifice_head, "\n")
    #print("pressure diffrence venturi\n",pressure_diff_venturi,"\n")
    #print(pressure_diff_diffuser, "\n")
    #print(pressure_diff_orifice, "\n")
    
    #print("Reynalds number venturi \n" ,Re_venturi)
    #print("Reynalds number orifice \n" ,Re_orifice)
    
    #print(venturi_cd)
    #print(orifice_cd)
    
    #print("venturi flow rate\n",venturi_flow_rate)
    #print(orifice_flow_rate)
    #print(rotometer_flow_rate)
    
    #print("error_pressure_diff_venturi\n", error_pressure_diff_venturi )
    #print("error_flow_rates\n", error_flow_rates)
    # print("error_venturi_area_1", error_area_1_venturi) 
    # print("error_venturi_area_2", error_area_2_venturi)
    # print("error_orifice_area_1", error_area_1_orifice )
    # print("error_orifice_area_2", error_area_2_orifice)
    # print("density_uncertantity", density_uncertantity)
    #print("error_flow_venturi_upper", error_flow_venturi_upper )
    #print("error_flow_venturi_lower", error_flow_venturi_lower)
    #print("error_flow_orifice_upper", error_flow_orifice_upper)
    #print("rotameter_uncertanty", rotameter_uncertanty)
    
    
main()

