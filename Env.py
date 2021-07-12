#import modules
import os
import copy
from os.path import normpath, join
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
import multiprocessing as mp

from System.Network_3ph_pf import Network_3ph
#import P2P_Ahead
import System.Assets as AS
import System.Markets as MK
import System.EnergySystem as ES
import System.P2P_Trading as P2P

class Env():

	def __init__(self, network_type='eulv_reduced', vpu_bus_max = 1.05,\
        		vpu_bus_min = 0.95, p_line_max = 2000):

		self.network_type = network_type
		self.vpu_bus_max = vpu_bus_max
		self.vpu_bus_min = vpu_bus_min
		self.p_line_max = vpu_bus_min

		np.random.seed(1000)

		dt_raw = 1/60
		T_raw = int(24/dt_raw) #Number of data time intervals
		dt = 30/60 #5 minute time intervals
		T = int(24/dt) #Number of intervals
		dt_ems = 30/60 #5 minute time intervals
		T_ems = int(24/dt_ems) #Number of intervals
		dt_rt = 30/60 
		T_rt = int(24/dt_rt) #Number of intervals
		T0 = 0 #from 8 am to 8 am
		T0_loads = 8-T0

		#wholesale market
		prices_wsm = pd.read_csv('half-hourly-wholesale-prices-MWh-29-06-2021.csv', delimiter='\t')
		prices_wsm = np.array(prices_wsm)  #prices £/MWh
		dt_wsm = 24/len(prices_wsm)
		T_wsm = len(prices_wsm)
		prices_wsm_ems = np.zeros((T_ems,2)) # col0 for day ahead, col1 for intraday
		if dt_ems <= dt_wsm:
			for t in range(T_wsm):
				prices_wsm_ems[t*int(dt_wsm/dt_ems) : (t+1)*int(dt_wsm/dt_ems),:] = prices_wsm[t,:]/1e3
		else:
			for t in range(T_ems):
				prices_wsm_ems[t,:] = np.mean(prices_wsm[t*int(dt_ems/dt_wsm) : (t+1)*int(dt_ems/dt_wsm),:], axis=0)
				prices_wsm_ems[t,:] =prices_wsm_ems[t,:]/1e3

		print('**********************************************')
		print('* Initializing the network')
		print('**********************************************')

		#######################################
		### STEP 0: Load the data
		#######################################
		Loads_data_path = os.path.join("Data", "Loads_1min.csv")    
		Loads_raw = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values
		N_loads_raw = Loads_raw.shape[1]
		#aggregate data from a granularity of dt_raw = 1min to dt=30 min
		Loads = Loads_raw.transpose().reshape(-1,int(dt/dt_raw)).sum(1).reshape(N_loads_raw,-1).transpose()
		#print('load', Loads[:3,:3])
		Load_ems = Loads.transpose().reshape(-1,int(dt_ems/dt)).sum(1).reshape(N_loads_raw,-1).transpose()

		PV_data_path = os.path.join("Data", "PV_profiles_norm_1min.txt") 
		PVpu_raw = pd.read_csv(PV_data_path, sep='\t').values 
		PVpu_raw = PVpu_raw[:,:55]
		PVpu_raw = np.vstack((PVpu_raw,np.zeros(55)))
		N_pv_raw = PVpu_raw.shape[1]
		PVpu = PVpu_raw.transpose().reshape(-1,int(dt/dt_raw)).sum(1).reshape(N_pv_raw,-1).transpose()
		#print('PVpu:', PVpu[:5,:5])
		PVpu_ems = PVpu.transpose().reshape(-1,int(dt_ems/dt)).sum(1).reshape(N_pv_raw,-1).transpose()
		#take one pv profile and multiply it after with nominal power of each home. Reasonable since all houses are in the
		# same area. To account for the very short term disturbances (clouds), we can add a noise 
		PVpu = PVpu[:,0] 
		#print('PVpu:',PVpu)
		#######################################
		### STEP 1: setup parameters
		######################################
		N_loads = 3 #55

		###Community PV generation parameters
		"""N_cpv = 2
		cpv_bus_names = ['226','839','169']
		cpv_locs = [55,56,57]
		P_cpv_nom = 60 #maximum EV charging power"""

		#Home PV parameters
		N_pvbatt = 2 #40
		Ppv_home_nom = 4 #power rating of the PV generation (can be=np.random.randint(1, 15, size=N_pvbatt))
		#among N_loads inhabitants, select random N_pvbatt candidates who will act as prosumers (having pv+batt)
		pv_load_locs = np.random.choice(N_loads, N_pvbatt, replace=False)
		Pbatt_max = 4 #we can have vector here of size N_pvbatt 
		self.Ebatt_max = 8
		self.c1_batt_deg = 0.05 #Battery degradation costs


		#######################################
		### STEP 2: setup the network
		#######################################
		network = Network_3ph() 
		if self.network_type == 'eulv_reduced':
			network.setup_network_eulv_reduced()
		else:
			network.loadDssNetwork(self.network_type)		
		N_buses = network.N_buses
		N_phases = network.N_phases
		
		# set bus voltage limits
		network.set_pf_limits(self.vpu_bus_min*network.Vslack_ph, self.vpu_bus_max*network.Vslack_ph,
		                      self.p_line_max*1e3/network.Vslack_ph)

		#buses that contain loads
		load_buses = np.where(np.abs(network.bus_df['Pa'])+np.abs(network.bus_df['Pb'])+np.abs(network.bus_df['Pc'])>0)[0]
		print('load buses of the first 3 homes', load_buses)
		#contain the phases of each load, indexed on bus...
		#load_phase[0] lists the phases of the bus0 to which the load is connected
		load_phases = []
		N_load_bus_phases=0
		for i in range(len(load_buses)):
			phases = []
			if np.abs(network.bus_df.iloc[load_buses[i]].Pa) > 0:
				phases.append(0)
			if np.abs(network.bus_df.iloc[load_buses[i]].Pb) > 0:
				phases.append(1)
			if np.abs(network.bus_df.iloc[load_buses[i]].Pc) > 0:
				phases.append(2)
			load_phases.append(np.array(phases))
			N_load_bus_phases += len(phases)  #total of bus_phase connected
		print('phases of the 2 first loads', load_phases)

		load_phases = np.array([load_phases[i] for i in range(len(load_phases)) if len(load_phases[i])==1])
		load_buses = np.array([load_buses[i] for i in range(len(load_phases)) if len(load_phases[i])==1])
		# add the communal pvs t the lists
		"""for i in range(N_cpv):
			load_buses = np.append(load_buses,np.where(network.bus_df['name']==cpv_bus_names[i])[0][0])
			load_phases.append(np.arange(N_phases)) #cpv are thriphase
			N_load_bus_phases += N_phases """

		print('N buses= ', network.N_buses)
		N_load_buses = load_buses.size
		print('N load buses', N_load_buses)
		N_lines = network.N_lines
		N_line_phases = N_lines*N_phases

		#list of placement bus_phase that are not zero (will be needed for the nw_pf simulation)
		load_bus_phases = np.zeros(N_load_bus_phases)
		load_bus_phase_count = 0
		for i in range(N_load_buses):
			bus = load_buses[i]
			for ph_idx in range(len(load_phases[i])):
				load_bus_phases[load_bus_phase_count] = 3*bus + load_phases[i][ph_idx]
				load_bus_phase_count += 1

		load_bus_phases = load_bus_phases.astype(int)
		constrained_lines = []
		"""
		constrained_load_bus_names = ['249','225','264','562','619','676','899','906','835']
		constrained_load_bus_ph =[1,0,2,0,2,1,1,0,2]
		constrained_load_bus_phases = np.zeros(len(constrained_load_bus_names),dtype=int)
		for i in range(len(constrained_load_bus_names)):
			bus_name = constrained_load_bus_names[i]
			bus = np.where(network.bus_df['name']==bus_name)[0][0]
			constrained_load_bus_phases[i] = 3*bus + constrained_load_bus_ph[i]
		"""
		#######################################
		### STEP 3: setup the assets 
		#######################################
		storage_assets = []
		nondispatch_assets = []

		#55 Homes
		self.Loads_actual = Loads[:,0:N_loads]
		for i in range(N_loads):
			Pnet = self.Loads_actual[:,i]
			Qnet = self.Loads_actual[:,i]*0.05
			load_i = AS.NondispatchableAsset_3ph(Pnet, Qnet, load_buses[i], load_phases[i], dt, T)
			load_i.Pnet_pred = load_i.Pnet
			load_i.Qnet_pred = load_i.Qnet
			nondispatch_assets.append(load_i) 
		
		for i in range(N_loads):
			if i in pv_load_locs:
				Pmax_pv_i = np.zeros(T_ems)
				Pmin_pv_i = -PVpu*Ppv_home_nom #- PVpu [np.where(pv_load_locs ==i)] * Ppv_home_nom[np.where(pv_load_locs ==i)]
				pv_i = AS.PV_Controllable_3ph(Pmax_pv_i,Pmin_pv_i, load_buses[i], load_phases[i], dt, T, dt_ems, T_ems)
				pv_i.Pnet_pred = pv_i.Pmin
				storage_assets.append(pv_i)
		    
		# Home batteries: add a battery to every house having pv
		for i in range(N_pvbatt): 
			bus_id_i = load_buses[pv_load_locs[i]]
			phases_i = load_phases[pv_load_locs[i]]
			Emax_i = self.Ebatt_max*np.ones(T_ems)
			Emin_i = np.zeros(T_ems)
			ET_i = self.Ebatt_max*0.5
			E0_i = self.Ebatt_max*0.5        
			Pmax_i = Pbatt_max*np.ones(T_ems)
			Pmin_i = -Pbatt_max*np.ones(T_ems)
			batt_i = AS.StorageAsset_3ph(Emax_i, Emin_i, Pmax_i, Pmin_i, E0_i, ET_i, bus_id_i, phases_i, dt, T, dt_ems, T_ems,c_deg_lin = self.c1_batt_deg)
			storage_assets.append(batt_i)
		
		#30 Home EVs
		#EV_indexes = np.zeros(N_EVs)
		#es_count = 0

		#Community based ES systems
		"""for i in range(N_cpv):
			Pmax_cpv_i = np.zeros(T_ems)
			Pmin_cpv_i = -PVpu*P_cpv_nom
			bus_id_cbs_i = load_buses[cpv_locs[i]]
			phases_i = load_phases[cpv_locs[i]]
			cpv_i = AS.PV_Controllable_3ph(Pmax_cpv_i,Pmin_cpv_i, bus_id_cbs_i, phases_i, dt, T, dt_ems, T_ems)
			cpv_i.Pnet_pred = -PVpu_pred*P_cpv_nom
			storage_assets.append(cpv_i)"""
		   
		print('****************************************************')
		print('* Network is set, now linking prosumers to assets...')
		print('****************************************************')        

		#flag_P2P = True

		"""    
		#NO P2P
		if flag_P2P == False:
			save_id_string = save_id_string + '_NoP2P_'
		else:
			save_id_string = save_id_string + '_P2P_'

		save_id_string = save_id_string  + '_postcurt'
		#save_id_string = save_id_string + str(EditNum) + '_postcurt'
		"""

		################
		### P2P Parameters
		################
		N_pro_pvbatt = N_pvbatt
		N_pro_load = N_loads - N_pro_pvbatt   #pure consumers

		pro_pvbatt_bus_locs = load_buses[pv_load_locs]
		pro_load_bus_loc = np.array([i for i in load_buses[0:N_loads] if i not in pro_pvbatt_bus_locs])

		#bus indexes of all customers (prosumers nad pure consumers)
		self.pro_pvbatt_indexes = np.arange(0,N_pro_pvbatt)
		self.pro_load_indexes = np.arange(N_pro_pvbatt,N_pro_pvbatt+N_pro_load)
		#self.pro_cpv_indexes = np.arange(N_pro_pvbatt+N_pro_load,N_pro_pvbatt+N_pro_load+N_cpv)

		self.trade_energy = 1.0*dt_ems   #0.25 | 1
		self.price_inc = 0.05            #0.02 | 0.1    #price increment for buying deltaQ of energy  	
		self.trade_energy_rt = self.trade_energy*dt_rt/dt_ems

		#Number of separate P2P energy trading networks
		self.N_p2p_groups = 1

		################
		### Setup P2P Agents
		################

		agents = []
		pro2as = []

		agent_index = 0

		nopv_load_locs = np.array([i for i in range(N_loads) if i not in pv_load_locs])


		#storage_assets = self.energy_system.storage_assets

		for pro_pvbatt_index in range(N_pro_pvbatt):
			#Reminder pv_load_locs contains 40 elements from 55, it's just list of indices. 
			#For the bus do: load_bus[bus_id_i]
			# for the phase: load_phases[bus_id_i]
			#For the placement in the 3N x 3N matrix: [3*load_bus[bus_id_i] + ph for ph in load_phases[bus_id_i]]
			bus_id_i = pv_load_locs[pro_pvbatt_index] #With respect to load bus phases, ie load_buses[bus_id_i]
			Pnet_i = Loads[:,pv_load_locs[pro_pvbatt_index]] - PVpu*Ppv_home_nom 
			Ppv_i = PVpu*Ppv_home_nom
			#we're adding 40 to the index because the first elements of the list are pv connected to homes 
			storage_asset_index = N_pvbatt + pro_pvbatt_index
			Pch_max_i = -storage_assets[storage_asset_index].Pmin
			for t in range(T_ems):
				if PVpu[t]<=1e-4:
					Pch_max_i[t] = 0  
			Pdis_max_i = storage_assets[storage_asset_index].Pmax
			Pch_min_i = np.zeros(T_ems)
			Pdis_min_i = np.zeros(T_ems)
			Emax_flex_i = storage_assets[storage_asset_index].Emax
			Emin_flex_i = storage_assets[storage_asset_index].Emin
			E0_flex_i = storage_assets[storage_asset_index].E0
			ET_flex_i = storage_assets[storage_asset_index].ET
			c1_batt_deg_i = storage_assets[storage_asset_index].c_deg_lin
			flex_type_i = 'battery'
			placement_i = [3*load_buses[bus_id_i] + ph for ph in load_phases[bus_id_i]]
			p2p_group_i = agent_index%self.N_p2p_groups
			pro_i = P2P.Prosumer(agent_index,bus_id_i,Pnet_i,Pch_max_i,Pch_min_i,\
		                         Pdis_max_i,Pdis_min_i,Emax_flex_i,Emin_flex_i,E0_flex_i,ET_flex_i,flex_type_i,\
		                         np.zeros(T_ems), np.zeros(T_ems),\
		                         dt,T,dt_ems,T_ems, c1_deg=c1_batt_deg_i, p2p_group = p2p_group_i,Ppv = Ppv_i, \
		                         Ppv_ahead = Ppv_i, Pnet_pred_ahead = Pnet_i)

			agents.append(pro_i) 
			pro2as.append({'storage_asset': storage_asset_index,
						   'nondispatch_asset': pv_load_locs[pro_pvbatt_index]})
			#pro_pvbatt_id_list.append(agent_index)
			agent_index += 1

		for pro_load_index in range(N_pro_load):
			bus_id_i = nopv_load_locs[pro_load_index] #With respect to load bus phases
			Pnet_i = Loads[:,nopv_load_locs[pro_load_index]]
			Pch_max_i = np.zeros(T_ems)
			Pdis_max_i = np.zeros(T_ems)
			Pch_min_i = np.zeros(T_ems)
			Pdis_min_i = np.zeros(T_ems)
			Emax_flex_i = np.zeros(T_ems)
			Emin_flex_i = np.zeros(T_ems)
			E0_flex_i = 0
			ET_flex_i = 0
			flex_type_i = 'none'
			placement_i = [3*load_buses[bus_id_i] + ph for ph in load_phases[bus_id_i]]
			p2p_group_i = agent_index%self.N_p2p_groups
			pro_i = P2P.Prosumer(agent_index,bus_id_i,Pnet_i,Pch_max_i,Pch_min_i,\
			                     Pdis_max_i,Pdis_min_i,Emax_flex_i,Emin_flex_i,E0_flex_i,ET_flex_i,flex_type_i,\
			                    np.zeros(T_ems), np.zeros(T_ems),\
			                    dt,T,dt_ems,T_ems, p2p_group = p2p_group_i, Pnet_pred_ahead = Pnet_i)
			agents.append(pro_i)
			pro2as.append({'storage_asset': None,
						   'nondispatch_asset': nopv_load_locs[pro_load_index]})
			#pro_loadonly_id_list.append(agent_index)
			agent_index += 1
		    
		"""for pro_cpv_index in range(N_cpv):
		#!!!!!!!!!!!!!!cpv_loc ([55,56,57]) is an index on the buses vector bus_cpv = loads_bus[cpv_loc]
			bus_id_i = cpv_locs[pro_cpv_index]  #+2*pro_cpv_index #55, 58, 61
			Ppv_i = PVpu*P_cpv_nom
			Pnet_i = - Ppv_i	  
			Pnet_pred_rt_i = Pnet_i
			Ppv_ahead_i = np.zeros(T_ems)
			for t in range(T_ems):
				t_indexes = (t*dt_ems/dt + np.arange(dt_ems/dt)).astype(int)
				Ppv_ahead_i[t] = np.mean(PVpu_pred[t_indexes]*P_cpv_nom)
			Pnet_pred_ahead_i = - Ppv_ahead_i 				 
			Pch_max_i = np.zeros(T_ems)
			Pdis_max_i = np.zeros(T_ems)
			Pch_min_i = np.zeros(T_ems)
			Pdis_min_i = np.zeros(T_ems)
			Emax_flex_i = np.zeros(T_ems)
			Emin_flex_i = np.zeros(T_ems)
			E0_flex_i = 0
			ET_flex_i = 0
			flex_type_i = 'none'
			p2p_group_i = agent_index%self.N_p2p_groups
			placement_i = [3*load_buses[bus_id_i] + ph for ph in load_phases[bus_id_i]]   	    
			pro_i = P2P.Prosumer(agent_index,bus_id_i,Pnet_i,Pnet_pred_rt_i,Pnet_pred_ahead_i,Pch_max_i,Pch_min_i,\
			                     Pdis_max_i,Pdis_min_i,Emax_flex_i,Emin_flex_i,E0_flex_i,ET_flex_i,flex_type_i,\
			                     np.mean(self.prices_buy[:,placement_i],1), np.mean(self.prices_sell[:,placement_i],1),\
			                     dt,T,dt_ems,T_ems, p2p_group = p2p_group_i,Ppv = Ppv_i,Ppv_ahead = Ppv_ahead_i,phases=np.array([0,1,2]))
			agents.append(pro_i)
			pro2as.append({'storage_asset': None,
						'nondispatch_asset': cpv_locs[pro_cpv_index] })
			agent_index += 1"""

		self.N_agents = agent_index
		self.state_shape = (self.N_agents,3) #= (self.N_phases*self.N_agents,3) for 3phase connection

		self.dt_raw = dt_raw
		self.T_raw = T_raw #Number of data time intervals
		self.dt = dt #5 minute time intervals
		self.T = T #Number of intervals
		self.dt_ems = dt_ems #5 minute time intervals
		self.T_ems = T_ems #Number of intervals
		self.dt_rt = dt_rt
		self.T_rt = T_rt #Number of intervals
		self.T0 = T0 #from 8 am to 8 am
		self.T0_loads = T0_loads

		self.N_loads = N_loads
		self.N_phases = N_phases
		self.N_buses = N_buses		
		self.pv_load_locs = pv_load_locs
		self.agents = agents
		self.load_buses = load_buses
		self.load_phases = load_phases
		self.PVpu = PVpu
		self.Ppv_home_nom =Ppv_home_nom
		#self.P_cpv_nom = P_cpv_nom
		self.pro2as = pro2as

		self.load_bus_phases = load_bus_phases
		#self.constrained_load_bus_phases = constrained_load_bus_phases
		self.constrained_lines = constrained_lines
		self.network = network

		self.price_0 = np.zeros((self.T_ems, self.state_shape[0],2)) 
		for t in range(self.T_ems):
			self.price_0[t,:,0] = prices_wsm_ems[t,0]
			self.price_0[t,:,1] = prices_wsm_ems[t,1]

		print('**********************************************************')
		print('* Network is set, Prosumers identified, Start learning now')
		print('**********************************************************') 

	def normalisation(self, state):
		Pnet_max = np.amax(self.Loads_actual)
		Pnet_min = np.amin(self.Loads_actual) - self.Ppv_home_nom
		Price_min = np.amin(self.price_0)
		Price_max = np.amax(self.price_0)
		#print(Pnet_min, Pnet_max)
		state[:,:,0] = (state[:,:,0] - Pnet_min) / (Pnet_max - Pnet_min)
		state[:,:,1] = state[:,:,1] / self.Ebatt_max
		state[:,:,2] = (state[:,:,2] - Price_min) / (Price_max - Price_min)

		return state


	def noised_signal(self,df, t, sigma):
		df_temp =copy.deepcopy(df)
		df_temp[t] = df_temp[t] + np.random.normal(0, sigma)
		return df_temp

	def reset(self):
		self.episode_step=0

		state_variables = np.zeros((self.T_ems, self.state_shape[0], 3))  #Pnet,ET, price_0 (maybe Emax and Emin too?Qnet is not included in P2P)
		for index, agent in enumerate(self.agents):
			## add a noise to the actual step. here step=0
			##It simulates the realistic scenario that the actual is slightly different from the predicted
			agent.Pnet = self.noised_signal(agent.Pnet_pred_ahead, self.episode_step, 0.1)
			state_variables[:, index, 0] = agent.Pnet
			state_variables[:, index, 1] = agent.E_flex_ems
			state_variables[:,:,2] = self.price_0[:,:,0]
		#print('state shape at reset', state_variables.shape)
		return self.normalisation(state_variables)

	def step(self, actions):
		##observation
		self.prices_buy, self.prices_sell = actions[:,:,0], actions[:,:,1]

		for i in range(len(self.agents)):
		#	placement = [3*self.load_buses[agent.bus_id] -3 + ph for ph in self.load_phases[agent.bus_id]]
		#	agent.sup_prices_buy, agent.sup_prices_sell = np.mean(actions[:,placement,0], 1), np.mean(actions[:,placement,1], 1) 
			self.agents[i].sup_prices_buy, self.agents[i].sup_prices_sell = actions[:,i,0], actions[:,i,1] 
		
		##Reward
		reward, violations, revenue, revenue_pro = self.reward(actions[:,:,2:]) 
		#Pnet, Ppv are the same as Pnet_pred_ahead, Ppv_ahead with two diff: 
		#### a circular shifting: at episode_step=t Pnet and Ppv are ordered [t, t+1, ...T, 0,...,(t-1)]
		#### the uncertainty to (Pnet, Ppv) is added as a noise for just the time step t


		self.episode_step +=1
		#next observation (roll P and update Et), update agents
		next_state = np.zeros((self.T_ems, self.state_shape[0], 3))  #Pnet,ET (maybe Emax and Emin too?Qnet is not included in P2P)
		for index, agent in enumerate(self.agents):
			## add a noise to the actual step. here step=0
			##It simulates the realistic scenario that the actual is slightly different from the predicted
			agent.Pnet = self.noised_signal(agent.Pnet_pred_ahead, self.episode_step, 0.1)
			agent.Pnet = np.roll(agent.Pnet, -self.episode_step, axis=0)
			agent.Ppv = np.roll(agent.Ppv_ahead, -self.episode_step, axis=0)
			agent.E_flex_ems = np.roll(agent.E_flex_ems, -1, axis=0)

			next_state[:, index, 0] = agent.Pnet
			next_state[:, index, 1] = agent.E_flex_ems

		#the price is day_ahead price for all time steps except episode_step which is equal to intaday price
		next_state[:,:,2] = self.price_0[:,:,0]
		next_state[self.episode_step,:,2] = self.price_0[self.episode_step,:,1]
		next_state[:,:,2]  = np.roll(next_state[:,:,2] , -self.episode_step, axis=0)
		print('1st prosumer of agents at t={} (next state observation), E={}, Ppv={}, Ppv_ahead={}' 
			.format(self.episode_step, self.agents[0].E_flex_ems, self.agents[0].Ppv, self.agents[0].Ppv_ahead))

		#done?
		done = False
		if self.episode_step == self.T_ems-1:
			done = True

		return self.normalisation(next_state), reward, violations, revenue, revenue_pro, done


	def reward(self, sup_trans_costs):
		print("*************Start running the day ahead optimization with P2P feature*************")

		################
		### Setup Trades
		################
		trades_id_col = 0
		trades_seller_col = 1
		trades_buyer_col = 2
		trades_sell_price_col = 3
		trades_buy_price_col = 4
		trades_time_col = 5
		trades_energy_col = 6
		trades_tcost_col = 7
		trades_optimal_col = 8

		# performance metrics
		max_ahead_iterations = 0 #the max iterations done to settle P2P prices 
		max_ahead_iter_duration = 0 #the prosumer that took the longest for the optimization pb
		################
		### Day Ahead Trading
		################    
		trade_list_ems = np.empty((0, 9)) #X the set of potential bilateral contracts
		t_ahead_0 = 0
		
		print('*********************************************************')
		print('* Initializing X the set of potential bilateral contracts')
		print('*********************************************************')

		#Add 1 trade between each PV generator and each flex load owner per time interval
		#Trange_trading = range(int(4/dt_ems),int(20/dt_ems))
		trade_index = 0
		N_trades_ahead_s2b = np.zeros([self.T_ems,self.N_agents,self.N_agents], dtype=int)

		N_p2p_ahead_max = 4 #max nbr of ahead transactions that a prosumer can make with a specific prosumer 

		for t in range(self.T_ems):
			if t%12 == 0:
				print('Interval: ' + str(t) + ' of ' + str(self.T_ems))
			
			for buyer_id in np.append(self.pro_pvbatt_indexes,self.pro_load_indexes): 
			#for buyer_id in self.pro_load_indexes:
				#for seller_id in np.append(self.pro_pvbatt_indexes,self.pro_cpv_indexes): 
				for seller_id in self.pro_pvbatt_indexes:
					if buyer_id != seller_id and self.agents[buyer_id].p2p_group == self.agents[seller_id].p2p_group:
						#N_trades_ahead_s2b[t,seller_id,buyer_id] = min(max(np.ceil(1.0*(agents[buyer_id].Pnet_pred_ahead[t] + agents[buyer_id].Pnet_forward[t] + agents[buyer_id].Pch_max[t])/(self.trade_energy/self.dt_ems)),0),\
						#                                         max(np.ceil(1.0*(-agents[seller_id].Pnet_pred_ahead[t] - agents[seller_id].Pnet_forward[t] + agents[seller_id].Pdis_max[t])/(self.trade_energy/self.dt_ems)),0))
						N_trades_ahead_s2b[t,seller_id,buyer_id] = min(max(np.ceil(1.0*(self.agents[buyer_id].Pnet[t] + self.agents[buyer_id].Pch_max[t])/(self.trade_energy/self.dt_ems)),0),\
						                                               max(np.ceil(1.0*(-self.agents[seller_id].Pnet[t] + self.agents[seller_id].Pdis_max[t])/(self.trade_energy/self.dt_ems)),0))
						N_trades_ahead_s2b[t,seller_id,buyer_id] = min(N_trades_ahead_s2b[t,seller_id,buyer_id],N_p2p_ahead_max)
						for trade_i in range(N_trades_ahead_s2b[t,seller_id,buyer_id]):
							buy_price = 0# in the paper refers to \lambda_x^b algo1
							sell_price = 0#in the paper refers to \lambda_x^s
							tcost = sup_trans_costs[t, seller_id, buyer_id]
							trade_list_ems = np.append(trade_list_ems,[[trade_index,seller_id,buyer_id,sell_price,buy_price,t,self.trade_energy,tcost, 0]],axis = 0)
							trade_index +=1
		N_trades = trade_index
		done = False
		#Price Adjustment

		start_time = time.time()         

		agent_pref_output = [None]*self.N_agents
		agent_elapsed_da = [ [] for i in range(self.N_agents) ]
		
		for group_index in range(self.N_p2p_groups):
			iteration = 0
			done=False
			p2p_group_agent_indexes = [i for i in range(self.N_agents) if self.agents[i].p2p_group == group_index]
			#indices=[p2p_group_agent_indexes[i] for i in range(len(p2p_group_agent_indexes))]
			while done == False:
				iteration +=1
				print('*********************************************\n*\n*\n*')
				print('* Day Ahead Trading Price Adjustment: ')
				print('* Iteration: ' + str(iteration) + '| P2P Group ' + str(group_index+1) + ' of ' + str(self.N_p2p_groups) )
				print('*\n*\n*\n*********************************************')
				done = True			                
				"""
				start = time.time()
				pool = mp.Pool(processes=4)
				#agent_pref_output = pool.map(self.job, self.agents)
				agent_pref_output = pool.starmap(P2P.Prosumer.get_preferred_trades_mi_curt_RL, [(agent, trade_list_ems, t_ahead_0 , self.T_ems) for agent in self.agents])
				#agent_pref_output = [pool.apply(self.agents[i].get_preferred_trades_mi_curt_RL, args=(trade_list_ems,0,self.T_ems)) 
				#                      for i in indices]
				end = time.time() - start
				max_ahead_iter_duration = end
				pool.close() 
				pool.join()
				"""
				
				for i in range(len(p2p_group_agent_indexes)):
					agent_idx = p2p_group_agent_indexes[i]
					elapsed_da = time.time() - start_time
					print('Iter.: ' + str(iteration) + ' | Agent: ' + str(i+1) + ' of ' + str(len(p2p_group_agent_indexes)) + ' | Elapsed: ' + str(np.round(elapsed_da)) + ' sec')
					agent_i_start_time = time.time() 
					agent_pref_output[agent_idx] = self.agents[agent_idx].get_preferred_trades_mi_curt_RL(trade_list_ems,0,self.T_ems)
					agent_i_elapsed = time.time() - agent_i_start_time
					agent_elapsed_da[i].append(agent_i_elapsed)
					if agent_i_elapsed >= max_ahead_iter_duration:
						max_ahead_iter_duration = agent_i_elapsed
				
				q_ds_full_list = np.zeros(N_trades)
				q_us_full_list = np.zeros(N_trades)
				#for agent_idx in range(N_agents):
				q_ds_full_list[:] = np.sum(np.array(agent_pref_output[agent_idx]['q_ds_full_list'])[:,0] for agent_idx in p2p_group_agent_indexes)
				q_us_full_list[:] = np.sum(np.array(agent_pref_output[agent_idx]['q_us_full_list'])[:,0] for agent_idx in p2p_group_agent_indexes)

				for trade_idx in range(N_trades):
					if q_us_full_list[trade_idx] > q_ds_full_list[trade_idx]:
						if trade_list_ems[trade_idx,trades_buy_price_col] > trade_list_ems[trade_idx,trades_sell_price_col]:
							trade_list_ems[trade_idx,trades_sell_price_col] += self.price_inc
						else:
							trade_list_ems[trade_idx,trades_buy_price_col] += self.price_inc
						done = False

				N_ahead_iterations = iteration
				if N_ahead_iterations >= max_ahead_iterations:
					max_ahead_iterations = N_ahead_iterations
		
		#Get agent outcomes
		#p_net_ems = np.zeros([self.T_ems,self.N_agents])
		p_es_ch_ems = np.zeros([self.T_ems,self.N_agents])
		p_es_dis_ems = np.zeros([self.T_ems,self.N_agents])
		#p_net_exp_ems = np.zeros([self.T_ems,self.N_agent])

		for agent_idx in range(self.N_agents):
			#p_net_ems[:,agent_idx] = self.agents[agent_idx].P_net_ems
			p_es_ch_ems[t_ahead_0:,agent_idx] = np.array(agent_pref_output[agent_idx]['p_es_ch_val'])[:,0]
			p_es_dis_ems[t_ahead_0:,agent_idx] = np.array(agent_pref_output[agent_idx]['p_es_dis_val'])[:,0]

			trade_list_ems_b = trade_list_ems[trade_list_ems[:,trades_buyer_col]== agent_idx]
			trade_list_ems_b[:,trades_optimal_col] = np.array(agent_pref_output[agent_idx]['q_us_val']).reshape(-1)
			trade_list_ems[trade_list_ems[:,trades_buyer_col]== agent_idx] = trade_list_ems_b 
			#Update forward E of batteries as scheduled
			for t in range(t_ahead_0,self.T_ems):
				self.agents[agent_idx].update_flex_ems(t,p_es_ch_ems[t,agent_idx]-p_es_dis_ems[t,agent_idx])
			#agents[agent_idx].Pnet_forward[self.t_ahead_0:] += (1/self.dt_ems)*(q_us_ems[self.t_ahead_0:,agent_idx] - q_ds_ems[self.t_ahead_0:,agent_idx])
			#p_net_forward_ems[:,agent_idx] = agents[agent_idx].Pnet_forward
		print('q_imp={}, q_exp={}, trade_list_ems={}'.
			format(agent_pref_output[agent_idx]['q_sup_buy_val'][0], agent_pref_output[agent_idx]['q_sup_sell_val'][0],
					trade_list_ems ))
					
		print('*********************************************\n*\n*\n*')
		print('* P2P ahead acomplished, Computing the reward ')
		print('*\n*\n*\n*********************************************')    

		#for the reward, we have to update the assets of the energy_system 
		p_net_actual = np.zeros([self.T,self.N_agents])
		Pnet_load = np.zeros([self.T,self.N_agents])
		q_p2p_P2P_Ahead = np.zeros([self.T_ems,self.N_agents])

		network_pf = copy.deepcopy(self.network)
		network_pf.clear_loads()

		for agent_idx in range(self.N_agents):
			for t in range(self.T):
				t_ahead = int(np.floor(t*self.dt/self.dt_ems))
				p_net_actual[t,agent_idx] = self.agents[agent_idx].Pnet[t] + self.agents[agent_idx].P_flex_ems[t_ahead]
				if agent_idx in self.pv_load_locs:
					Pnet_load[t,agent_idx] = p_net_actual[t,agent_idx] + self.agents[agent_idx].Ppv[t]
				else:
					Pnet_load[t,agent_idx] = p_net_actual[t,agent_idx] 
			Pnet = p_net_actual[:,agent_idx]
			Qnet = Pnet_load[:,agent_idx]*0.05
			network_pf.set_load(self.load_buses[self.agents[agent_idx].bus_id], self.load_phases[self.agents[agent_idx].bus_id],
				                Pnet[0], Qnet[0])
		network_pf.zbus_pf()
		network_pf.linear_model_setup(network_pf.v_net_res, network_pf.S_PQloads_wye_res, network_pf.S_PQloads_del_res) 
		#note that phases need to be 120degrees out for good results
		network_pf.linear_pf()
		#print('ouf')


		#### DSO revenue #### G_wye shape (3, 756)
		"""
		p0 = np.zeros(self.N_phases)
		for agent in self.agents:
			bus = self.load_buses[agent.bus_id]
			phases = self.load_phases[agent.bus_id]
			P = (agent.Pnet[0] + agent.P_flex_ems[0])/len(phases) 
			for phase in phases:
				if network_pf.bus_df[network_pf.bus_df['number']==bus]['connect'].values[0]=='Y':
					p0 += np.real(network_pf.G_wye[:,3*bus + phase -3])*P
				else:
					p0 += np.real(network_pf.G_del[:,3*bus + phase -3])*P
		p0 = p0 +  np.real(network_pf.G0)
		
		p0 = np.real(network_pf.S_PQloads_wye_res)+np.real(network_pf.S_PQloads_del_res)
		"""
		p0 = np.zeros(self.N_phases)
		s0 = network_pf.res_bus_df.iloc[0]
		p0[0] = np.real(s0['Sa'])
		p0[1] = np.real(s0['Sb'])
		p0[2] = np.real(s0['Sc'])	
		print('p0',p0)

		ems_list_0 = trade_list_ems[trade_list_ems[:,trades_time_col]==0]
		print('ems_list_0',ems_list_0)
		print('q_us_0_pros0',agent_pref_output[0]['q_us_0'])
		print('q_ds_0_pros0',agent_pref_output[0]['q_ds_0'])

		revenue=np.multiply(ems_list_0[:,trades_optimal_col],ems_list_0[:,trades_tcost_col]).sum() \
		       - self.price_0[self.episode_step,0,1]*np.sum(p0)*self.dt_ems
		print('revenue1', revenue)
		for agent_idx, agent in enumerate(self.agents):
			revenue += (agent.sup_prices_buy[0]*agent_pref_output[agent_idx]['q_sup_buy_val'][0,0]) \
	         - (agent.sup_prices_sell[0]*agent_pref_output[agent_idx]['q_sup_sell_val'][0,0]) 
		print('revenue', revenue)

		buses_Vpu = np.zeros([self.N_buses*self.N_phases])
		for bus_id in range(self.N_buses):
			bus_res = network_pf.res_bus_df.iloc[bus_id]
			buses_Vpu[3*bus_id] = np.abs(bus_res['Va'])/self.network.Vslack_ph        
			buses_Vpu[3*bus_id+1] = np.abs(bus_res['Vb'])/self.network.Vslack_ph                  
			buses_Vpu[3*bus_id+2] = np.abs(bus_res['Vc'])/self.network.Vslack_ph
		
		A_vlim_wye = 1e3*network_pf.K_wye
		A_vlim_del = 1e3*network_pf.K_del
		#b_vlim = network_pf.v_lin_abs_res

		reward = np.ones((self.N_agents, self.N_agents+2))

		violations, stable = 0,0
		for bus_id in range(self.N_buses):
			for phase_id in range(3):

				if buses_Vpu[3*bus_id + phase_id -3] > self.vpu_bus_max:
					Vgap = self.vpu_bus_max - buses_Vpu[3*bus_id + phase_id -3]
					print('Vmax violation', Vgap)
					print('bus-phase:{}-{}'.format(bus_id, phase_id))
					violations+=1	
					for agent_idx, agent in enumerate(self.agents):
						bus = self.load_buses[agent.bus_id]
						phases = self.load_phases[agent.bus_id]
						#P = (agent.Pnet[0] + agent.P_flex_ems[0])/len(phases) 
						for phase in phases:

							if network_pf.bus_df[network_pf.bus_df['number']==bus]['connect'].values[0]=='Y':
								for k in range(len(self.agents)):
									q_us_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==agent_idx, ems_list_0[:,trades_seller_col]==k)]
									q_ds_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==k, ems_list_0[:,trades_seller_col]==agent_idx)]
									reward[agent_idx,2+k]+= A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*(np.sum(q_us_k[:,trades_optimal_col]) - np.sum(q_ds_k[:,trades_optimal_col]))*Vgap/self.T_ems
								#reward[agent_idx+phase,0] in case we consider three phase connection
								reward[agent_idx,0] += A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_buy_val'][0]*Vgap/self.T_ems
								reward[agent_idx,1] -= A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_sell_val'][0]*Vgap/self.T_ems
							else:
								for k in range(len(self.agents)):
									q_us_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==agent_idx, ems_list_0[:,trades_seller_col]==k)]
									q_ds_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==k, ems_list_0[:,trades_seller_col]==agent_idx)]
									reward[agent_idx,2+k] += A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*(np.sum(q_us_k[:,trades_optimal_col]) - np.sum(q_ds_k[:,trades_optimal_col]))*Vgap/self.T_ems
								reward[agent_idx,0] += A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_buy_val'][0]*Vgap/self.T_ems
								reward[agent_idx,1] -= A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_sell_val'][0]*Vgap/self.T_ems
										
				elif buses_Vpu[3*bus_id + phase_id -3] < self.vpu_bus_min:
					Vgap = self.vpu_bus_min - buses_Vpu[3*bus_id + phase_id -3]
					print('Vmin violation', Vgap)
					print('bus-phase:{}-{}'.format(bus_id, phase_id))
					violations+=1
					for agent_idx, agent in enumerate(self.agents):
						bus = self.load_buses[agent.bus_id]
						phases = self.load_phases[agent.bus_id]
						#P = (agent.Pnet[0] + agent.P_flex_ems[0])/len(phases)
						for phase in phases:

							if network_pf.bus_df[network_pf.bus_df['number']==bus]['connect'].values[0]=='Y':
								for k in range(len(self.agents)):
									q_us_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==agent_idx, ems_list_0[:,trades_seller_col]==k)]
									q_ds_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==k, ems_list_0[:,trades_seller_col]==agent_idx)]
									reward[agent_idx,2+k]+= A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*(np.sum(q_us_k[:,trades_optimal_col]) - np.sum(q_ds_k[:,trades_optimal_col]))*Vgap/self.T_ems
								reward[agent_idx,0] += A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_buy_val'][0]*Vgap/self.T_ems
								reward[agent_idx,1] -= A_vlim_wye[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_sell_val'][0]*Vgap/self.T_ems
							else:
								for k in range(len(self.agents)):
									#q_us_k = ems_list_0[ems_list_0[:,trades_buyer_col]==agent_idx and ems_list_0[:,trades_seller_col]==k]
									q_us_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==agent_idx, ems_list_0[:,trades_seller_col]==k)]
									q_ds_k = ems_list_0[np.logical_and(ems_list_0[:,trades_buyer_col]==k, ems_list_0[:,trades_seller_col]==agent_idx)]
									reward[agent_idx,2+k] += A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*(np.sum(q_us_k[:,trades_optimal_col]) - np.sum(q_ds_k[:,trades_optimal_col]))*Vgap/self.T_ems
								reward[agent_idx,0] += A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_buy_val'][0]*Vgap/self.T_ems
								reward[agent_idx,1] -= A_vlim_del[3*bus_id + phase_id -3, 3*bus + phase -3]*agent_pref_output[agent_idx]['q_sup_sell_val'][0]*Vgap/self.T_ems
				
				else:
					stable+=1

		print('reward imp=',reward[:,0])
		print('reward exp', reward[:,1])
		print('reward_P2P', reward[:, 2:])
		print('violations {} et stable {}'.format(violations, stable) )

		if revenue <0:  #first -437160, -417593, -429932
			reward = reward - 10
		else:
			reward = reward + 10

		revenue_pro = np.zeros((self.N_agents))
		for agent_idx, agent in enumerate(self.agents):
			ems_list_0_s = ems_list_0[ems_list_0[:,trades_seller_col]==agent_idx]
			ems_list_0_b = ems_list_0[ems_list_0[:,trades_buyer_col]==agent_idx]
			revenue_pro[agent_idx] = - (agent.sup_prices_buy[0]*agent_pref_output[agent_idx]['q_sup_buy_val'][0,0]) \
	         + (agent.sup_prices_sell[0]*agent_pref_output[agent_idx]['q_sup_sell_val'][0,0])
			+ np.multiply(ems_list_0_s[:,trades_energy_col], ems_list_0_s[:,trades_sell_price_col] - ems_list_0_s[:,trades_tcost_col]/2).sum() 
			- np.multiply(ems_list_0_b[:,trades_energy_col], ems_list_0_b[:,trades_sell_price_col] - ems_list_0_b[:,trades_tcost_col]/2).sum() 

		return reward, violations, revenue, revenue_pro
