############
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, '/eos/lhcb/user/h/havva/q2Measurement-BtoDstMuNu/GaussianProcessRegression/Packages/AmpliTF/')
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import uproot
from util_resolution import return_phasespace, get4mom, return_quadsolution, return_boostapprox
############

year = 2016#2016#2017,2018
magnets = ['up','down']
base_mc_path = f"/eos/lhcb/user/h/havva/q2Measurement-BtoDstMuNu/MCsamples/TrackerOnlyNTuples_Maryland/Angular_RDst_muonic_Run2/ReProcess_tuples/{year}"
# Components and file mapping
'''components = [
    "b0_dstmu"#, 
]'''
'''file_map = {
    "signal": "b0_dstmu/b0_dstmunu",
    "b0_dsttau": "b0_dsttau/b0_dsttaunu",
    "b0_dststtau": "b_dststtau/b0_dststtau",
    "b+_dststtau": "b_dststtau/b+_dststtau",
    "b+_dstdx": "DDX/b+_dstdx",
    "b0_dstdx": "DDX/b0_dstdx",
    "b0_dststmu": "b_dststmu/b0_dststmu",
    "b+_dststmu": "b_dststmu/b+_dststmu",
    "bs_dststmu": "bs_dststmu/bs_dststmu",
    "b0_dststmu_415": "b_dststmu/b0_dststmu",
    "b0_dststmu_10413": "b_dststmu/b0_dststmu",
    "b0_dststmu_20413": "b_dststmu/b0_dststmu",
    "b+_dststmu_425": "b_dststmu/b+_dststmu",
    "b+_dststmu_10423": "b_dststmu/b+_dststmu",
    "b+_dststmu_20423": "b_dststmu/b+_dststmu",
    "bs_dststmu_10433": "bs_dststmu/bs_dststmu",
    "bs_dststmu_435": "bs_dststmu/bs_dststmu",
    "b+_dststmu_heavy": "b_dststmu_heavy/b+_dststmu_heavy",
    "b0_dststmu_heavy": "b_dststmu_heavy/b0_dststmu_heavy"
}
'''
selected_branches = [
    "b0_PX", "b0_PY", "b0_PZ", "b0_PE","b0_P","b0_PT",
    "mu_PX", "mu_PY", "mu_PZ", "mu_PE", "mu_P","mu_PT",
    "dst_PX", "dst_PY", "dst_PZ", "dst_PE", "dst_P","dst_PT",
    "d0_PX", "d0_PY", "d0_PZ", "d0_PE", "d0_PT", "d0_P",
    "spi_PX", "spi_PY", "spi_PZ", "spi_PE", "spi_P","spi_PT",
    "k_PX", "k_PY","k_PZ", "k_PE","k_P","k_PT", 
    "pi_PX", "pi_PY", "pi_PZ", "pi_PE", "pi_P","pi_PT",
    "b0_TRUEP_X", "b0_TRUEP_Y", "b0_TRUEP_Z", "b0_TRUEP_E","b0_TRUEPT",
    "pi_ETA", "k_ETA", "mu_ETA",
    #"b0_TRUEENDVERTEX_X", "b0_TRUEENDVERTEX_Y", "b0_TRUEENDVERTEX_Z",
    #"b0_TRUEORIGINVERTEX_X", "b0_TRUEORIGINVERTEX_Y", "b0_TRUEORIGINVERTEX_Z",
    "mu_TRUEP_X", "mu_TRUEP_Y", "mu_TRUEP_Z", "mu_TRUEP_E","mu_TRUEPT",
    "dst_TRUEP_X", "dst_TRUEP_Y", "dst_TRUEP_Z", "dst_TRUEP_E","dst_TRUEPT",
    "d0_TRUEP_X", "d0_TRUEP_Y", "d0_TRUEP_Z", "d0_TRUEP_E","d0_TRUEPT",
    "spi_TRUEP_X", "spi_TRUEP_Y", "spi_TRUEP_Z", "spi_TRUEP_E","spi_TRUEPT",
    "k_TRUEP_X", "k_TRUEP_Y", "k_TRUEP_Z", "k_TRUEP_E","k_TRUEPT",
    "pi_TRUEP_X", "pi_TRUEP_Y", "pi_TRUEP_Z","pi_TRUEPT",
    "b0_ENDVERTEX_X", "b0_ENDVERTEX_Y", "b0_ENDVERTEX_Z","b0_DIRA_OWNPV",
    "b0_OWNPV_X", "b0_OWNPV_Y", "b0_OWNPV_Z","b0_IP_OWNPV","b0_IPCHI2_OWNPV", 
    "b0_FDCHI2_OWNPV", "b0_FD_OWNPV", "b0_OWNPV_CHI2", "b0_OWNPV_NDOF",
    "k_IP_OWNPV","k_IPCHI2_OWNPV",
    "pi_IP_OWNPV", "pi_IPCHI2_OWNPV",
    "d0_IP_OWNPV", "d0_IPCHI2_OWNPV", "d0_FDCHI2_OWNPV","d0_FD_OWNPV", 
    "d0_DIRA_ORIVX","d0_DIRA_OWNPV","d0_ORIVX_CHI2", "d0_ORIVX_NDOF",
    "d0_OWNPV_CHI2","d0_OWNPV_NDOF",
    "dst_FDCHI2_OWNPV", "dst_IPCHI2_OWNPV", "dst_IP_OWNPV",
    "mu_IP_OWNPV","mu_IPCHI2_OWNPV",
    "dst_M", "b0_M", "d0_M",
    "b0_ISOLATION_BDT",
    "b0_TRUEID", "dst_TRUEID", "d0_TRUEID", "spi_TRUEID", "mu_TRUEID","mu_MC_MOTHER_ID",
    "b0_TrueHadron_D0_ID","b0_TrueHadron_D1_ID","b0_TrueHadron_D2_ID",
    "b0_TrueHadron_D0_PE","b0_TrueHadron_D0_PX","b0_TrueHadron_D0_PY","b0_TrueHadron_D0_PZ",
    "mu_MC_MOTHER_TRUEPE","mu_MC_MOTHER_TRUEPX","mu_MC_MOTHER_TRUEPY","mu_MC_MOTHER_TRUEPZ",
    "b0_TrueHadron_D1_PE","b0_TrueHadron_D1_PX","b0_TrueHadron_D1_PY","b0_TrueHadron_D1_PZ",
    "b0_TrueHadron_D2_PE","b0_TrueHadron_D2_PX","b0_TrueHadron_D2_PY","b0_TrueHadron_D2_PZ",
    "b0_TrueTauNuTau_PE","b0_TrueTauNuTau_PX","b0_TrueTauNuTau_PY","b0_TrueTauNuTau_PZ",
    "b0_TrueTauNuMu_PE","b0_TrueTauNuMu_PX","b0_TrueTauNuMu_PY","b0_TrueTauNuMu_PZ",
    "b0_TrueNeutrino_PE","b0_TrueNeutrino_PX","b0_TrueNeutrino_PY","b0_TrueNeutrino_PZ",# From this line on: those might not be necessary!
    "b0_TrueMu_PE","b0_TrueMu_PX","b0_TrueMu_PY","b0_TrueMu_PZ",
    "b0_TrueTau_PE","b0_TrueTau_PX","b0_TrueTau_PY","b0_TrueTau_PZ",
    "nTracks","runNumber","eventNumber"
]


selected_branches += ['nu_TRUEP_E','nu_TRUEP_X','nu_TRUEP_Y','nu_TRUEP_Z',
    # True fit vars
    'costhetal_TRUE','costhetad_TRUE','chi_TRUE','q2_TRUE','mmiss2_TRUE','w_TRUE',
    'nu_PE','nu_PX','nu_PY','nu_PZ',
    'b0_Flight_X','b0_Flight_Y','b0_Flight_Z',
    # Reco fit vars
    'q2rec_quad','costhetalrec_quad','costhetadrec_quad','chirec_quad',
    'mmiss2rec_brf','q2rec_plus','q2rec_minus','w_rec_quad',
    # Weight branches 
    'pid_eff'
]

# True masses
mB = 5279.61000000
mDst = 2010.27000000
ml  = 105.65837121

#define aliases
scale_theta = 0.32573304165853184
scale_phi   = 0.3769516994536387

aliases = {}
aliases['b0_Flight_R'] = '((b0_ENDVERTEX_X - b0_OWNPV_X)**2 + (b0_ENDVERTEX_Y - b0_OWNPV_Y)**2 + (b0_ENDVERTEX_Z - b0_OWNPV_Z)**2)**0.5'
aliases['b0_Flight_Theta'] = 'atan2(((b0_ENDVERTEX_X - b0_OWNPV_X)**2 + (b0_ENDVERTEX_Y - b0_OWNPV_Y)**2)**0.5, (b0_ENDVERTEX_Z - b0_OWNPV_Z))'
aliases['b0_Flight_Phi'] = 'atan2((b0_ENDVERTEX_Y - b0_OWNPV_Y), (b0_ENDVERTEX_X - b0_OWNPV_X))'
aliases['b0_Flight_X'] = 'b0_Flight_R * sin(b0_Flight_Theta) * cos(b0_Flight_Phi)'
aliases['b0_Flight_Y'] = 'b0_Flight_R * sin(b0_Flight_Theta) * sin(b0_Flight_Phi)'
aliases['b0_Flight_Z'] = 'b0_Flight_R * cos(b0_Flight_Theta)'


component = "b0_dstmunu"
#for component in components:
for mag in magnets:
#    files  = [{f'{base_mc_path}/{file_map[component]}_{mag}_allVariables.root': 'DecayTree'}] 
    files  = [{f'{base_mc_path}/{component}_{mag}_allVariables.root': 'DecayTree'}] 
    #files  += [{f'{base_mc_path}/{file_map[component]}_down_allVariables.root': 'DecayTree'}] 

    df_base = (uproot.open(files[0])).arrays(expressions=list(set(selected_branches)), aliases = aliases, library='pd')#entry_stop = 20000,
    #df2 = (uproot.open(files[1])).arrays(expressions=list(set(selected_branches)), aliases = aliases, library='pd')
    #df_base  = pd.concat([df1, df2])

    #############
    #True variables
    Pdst_lab    = atfk.lorentz_vector(atfk.vector(df_base['dst_TRUEP_X'], df_base['dst_TRUEP_Y'] , df_base['dst_TRUEP_Z']), df_base['dst_TRUEP_E']) 
    Pl_lab     = atfk.lorentz_vector(atfk.vector(df_base['mu_TRUEP_X'], df_base['mu_TRUEP_Y'], df_base['mu_TRUEP_Z']), df_base['mu_TRUEP_E'])
    Pnu_lab    = atfk.lorentz_vector(atfk.vector(df_base['nu_TRUEP_X'], df_base['nu_TRUEP_Y'], df_base['nu_TRUEP_Z']), df_base['nu_TRUEP_E'])
    # D0 (or D+) from D*
    PD_lab = atfk.lorentz_vector(atfk.vector(df_base['d0_TRUEP_X'], df_base['d0_TRUEP_Y'], df_base['d0_TRUEP_Z']),df_base['d0_TRUEP_E'])
    # soft pion from D* → D π
    Ppi_lab = atfk.lorentz_vector(atfk.vector(df_base['spi_TRUEP_X'], df_base['spi_TRUEP_Y'], df_base['spi_TRUEP_Z']),df_base['spi_TRUEP_E'])
    Pb_lab    = Pdst_lab + Pl_lab + Pnu_lab

    #TRUE mom and fit vars
    df_base['b0_P_TRUE']       = atfk.p(atfk.spatial_components(Pb_lab)).numpy()
    '''df_base['Lepton_E_TRUE']   = atfk.time_component(Pl_lab).numpy()
    df_base['Lepton_P_TRUE']   = atfk.p(atfk.spatial_components(Pl_lab)).numpy()
    df_base['Lepton_PX_TRUE']  = atfk.spatial_components(Pl_lab)[:,0].numpy()
    df_base['Lepton_PY_TRUE']  = atfk.spatial_components(Pl_lab)[:,1].numpy()
    df_base['Lepton_PZ_TRUE']  = atfk.spatial_components(Pl_lab)[:,2].numpy()'''
    #q2, costhl, costhd, chi, El, MM2 since we are using mmiss2_brf for reco NO NEED FOR REGRESSION
    #def return_phasespace(Pb0_lab, Pdst_lab, PLepton_lab, PD_lab, Ppi_lab, Missmass2=True, convert_to_numpy=True):
    q2, costhl, costhd, chi,_ = return_phasespace(Pb_lab, Pdst_lab, Pl_lab, PD_lab,Ppi_lab,False, True)
    df_base['q2_TRUE'] = q2
    df_base['costhetal_TRUE']= costhl
    df_base['costhetad_TRUE']= costhd
    df_base['chi_TRUE'] = chi

    '''#1st mom and fit vars
    df_base['b0_P_TRUE_plus']=return_quadsolution(atfk.spatial_components(Pb_lab), mB**2, Pdst_lab, Pl_lab, solution = 'plus').numpy()
    Pb_lab_plus = get4mom(atfk.spatial_components(Pb_lab), df_base['b0_P_TRUE_plus'], mB)
    df_base['q2_TRUE_plus'], df_base['costhetal_TRUE_plus'], df_base['costhetad_TRUE_plus'], df_base['chi_TRUE_plus'] = return_phasespace(Pb_lab, Pdst_lab, Pl_lab, PD_lab,Ppi_lab,False, True)

    #2nd mom and fit vars
    df_base['b0_P_TRUE_minus']=return_quadsolution(atfk.spatial_components(Pb_lab), mB**2, Pdst_lab, Pl_lab, solution = 'minus').numpy()
    Pb_lab_minus = get4mom(atfk.spatial_components(Pb_lab), df_base['b0_P_TRUE_minus'], mB)
    df_base['q2_TRUE_minus'], df_base['costhetal_TRUE_minus'], df_base['costhetad_TRUE_minus'],df_base['chi_TRUE_minus'] = return_phasespace(Pb_lab, Pdst_lab, Pl_lab, PD_lab,Ppi_lab,False, True)

    '''
    ######################
    #Reco variables
    ## b0_P_ is spatial momentum
    p3_Bfd      = atfk.vector(df_base['b0_Flight_X'], df_base['b0_Flight_Y'], df_base['b0_Flight_Z'])
    Pdst_lab_r    = atfk.lorentz_vector(atfk.vector(df_base['dst_PX'], df_base['dst_PY'] , df_base['dst_PZ']), df_base['dst_PE']) 
    PLepton_lab_r= atfk.lorentz_vector(atfk.vector(df_base['mu_PX'], df_base['mu_PY'],  df_base['mu_PZ']), df_base['mu_PE'])
    # D0 (or D+) from D*
    PD_lab_r = atfk.lorentz_vector(atfk.vector(df_base['d0_PX'], df_base['d0_PY'], df_base['d0_PZ']),df_base['d0_PE'])
    # soft pion from D* → D π
    Ppi_lab_r = atfk.lorentz_vector(atfk.vector(df_base['spi_PX'], df_base['spi_PY'], df_base['spi_PZ']),df_base['spi_PE'])
    #Pb0_lab_r   = Pdst_lab_r + PLepton_lab
    '''df_base['Lepton_E_Reco']   = atfk.time_component(PLepton_lab_r).numpy()
    df_base['Lepton_P_Reco']   = atfk.p(atfk.spatial_components(PLepton_lab_r)).numpy()
    df_base['Lepton_PX_Reco']  = atfk.spatial_components(PLepton_lab_r)[:,0].numpy()
    df_base['Lepton_PY_Reco']  = atfk.spatial_components(PLepton_lab_r)[:,1].numpy()
    df_base['Lepton_PZ_Reco']  = atfk.spatial_components(PLepton_lab_r)[:,2].numpy()
    '''
    #1st mom and fit vars
    df_base['b0_P_plus']=return_quadsolution(p3_Bfd, mB**2, Pdst_lab_r, PLepton_lab_r, solution = 'plus').numpy()
    Pb_lab_plus_r = get4mom(p3_Bfd, df_base['b0_P_plus'], mB)
    q2_plus, costhetal_plus, costhetad_plus,chi_plus,_ = return_phasespace(Pb_lab_plus_r, Pdst_lab_r, PLepton_lab_r,PD_lab_r,Ppi_lab_r,False,True)
    
    df_base['q2_plus']= q2_plus
    df_base['costhetal_plus']= costhetal_plus 
    df_base['costhetad_plus']= costhetad_plus
    df_base['chi_plus'] = chi_plus

    #2nd mom and fit vars
    df_base['b0_P_minus']=return_quadsolution(p3_Bfd, mB**2, Pdst_lab_r, PLepton_lab_r, solution = 'minus').numpy()
    Pb_lab_minus_r = get4mom(p3_Bfd, df_base['b0_P_minus'], mB)
    q2_minus, costhetal_minus, costhetad_minus,chi_minus,_ = return_phasespace(Pb_lab_minus_r, Pdst_lab_r, PLepton_lab_r,PD_lab_r,Ppi_lab_r,False,True)
    
    df_base['q2_minus']= q2_minus
    df_base['costhetal_minus']= costhetal_minus
    df_base['costhetad_minus']= costhetad_minus
    df_base['chi_minus'] = chi_minus

    ## this is from return_quadsolution and if there is Nan solution in Bp that event gets deleted!!!
    #store reco vars 
    cond              = (df_base['q2rec_quad'].values == df_base['q2rec_plus'].values)
    df_base['b0_P_Reco']   = np.where(cond, df_base['b0_P_plus'].values,     df_base['b0_P_minus'].values)
    df_base['q2_Reco']     = np.where(cond, df_base['q2_plus'].values,       df_base['q2_minus'].values)
    df_base['w_Reco']     = (mB**2 + mDst**2 - df_base['q2_Reco']) / (2 * mB * mDst)
    df_base['costhetal_Reco'] = np.where(cond, df_base['costhetal_plus'].values, df_base['costhetal_minus'].values)
    df_base['costhetad_Reco'] = np.where(cond, df_base['costhetad_plus'].values, df_base['costhetad_minus'].values)
    df_base['chi_Reco']       = np.where(cond, df_base['chi_plus'].values,       df_base['chi_minus'].values)

    '''## This is REST FRAME APPROXIMATION which we only use for mmiss2 and WE DO NOT NEED NOW !!!!!
    p_b0_mu      = return_boostapprox(p3_Bfd, (Pdst_lab_r + PLepton_lab_r), mB)
    PBmu_lab_r  = get4mom(p3_Bfd, p_b0_mu, mB)
    df_base['b0Mu_P_Reco'] = atfk.p(atfk.spatial_components(PBmu_lab_r)).numpy()
    df_base['q2Mu_Reco'], df_base['costhetalrec_brf'], df_base['costhetadrec_brf'], df_base['mmiss2rec_brf'] = return_phasespace(PBmu_lab_r, Pdst_lab_r, PLepton_lab_r, True)
    '''
    df_base = df_base.dropna() #Nan and negative sqrt solutions from return_quadsolution
    print(df_base.shape)
    ######################
    indx = range(df_base.shape[0])
    df_base.index    = indx
    df_base['Index'] = indx
    print(df_base)
    os.makedirs(f'./{component}', exist_ok=True)
    print('Storing', f'./{component}/preppedData_{year}_{mag}.p')
    df_base.to_pickle(f'./{component}/preppedData_{year}_{mag}.p')
