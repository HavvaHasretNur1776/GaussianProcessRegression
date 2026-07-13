###############
import sys, os
#home = os.getenv('HOME')
import numpy as np
sys.path.insert(0, '/eos/lhcb/user/h/havva/q2Measurement-BtoDstMuNu/GaussianProcessRegression/Packages/AmpliTF/')
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as patches

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
############### 

def MSE(y, y_pred): return np.mean((y - y_pred)**2, axis=0)

'''def get_fitvars(df, Pl_lab, p3_Bfd, mB, method):
    Pdst_lab   = atfk.lorentz_vector(atfk.vector(df['dst_PX'], df['dst_PY'] , df['dst_PZ']), df['dst_PE']) 
    df[f'b0_P_plus']  = return_quadsolution(p3_Bfd=p3_Bfd, m2_B=mB**2, p4_Dst=Pdst_lab, p4_mu=Pl_lab, solution= 'plus').numpy()
    df[f'b0_P_minus'] = return_quadsolution(p3_Bfd=p3_Bfd, m2_B=mB**2, p4_Dst=Pdst_lab, p4_mu=Pl_lab, solution= 'minus').numpy()
    #pick the solution that is closer to the prediction  
    cond1  = np.less(np.abs(df[f'b0_P_Estm'].values - df[f'b0_P_plus'].values),np.abs(df[f'b0_P_Estm'].values -  df[f'b0_P_minus'].values))    
    df[f'b0_P_Pred']  = np.where(cond1, df[f'b0_P_plus'].values, df[f'b0_P_minus'].values)
    #when two solutions are equal pick the prediction
    cond2  = np.equal(np.abs(df[f'b0_P_Estm'].values - df[f'b0_P_plus'].values) , np.abs(df[f'b0_P_Estm'].values - df[f'b0_P_minus'].values))
    df[f'b0_P_Pred']  = np.where(cond2, df[f'b0_P_Estm'].values, df[f'b0_P_Pred'].values)
    
    #calculate q2, costhl and mmiss2
    Pb0_lab   = get4mom(p3_Bfd, df[f'b0_P_Pred'], mB) 
    df[f'q2_Pred'],df[f'costhl_Pred'], _, df[f'mmiss2_Pred'] = return_phasespace(Pb0_lab, Pdst_lab, Pl_lab, True, True)
    df[f'q2_Pred']     = np.where(np.isnan(df[f'q2_Pred'].values), df['q2rec_quad'].values, df[f'q2_Pred'].values)
    df[f'costhl_Pred'] = np.where(np.isnan(df[f'costhl_Pred'].values), df['costhetalrec_quad'].values,df[f'costhl_Pred'].values)

    df[f'mmiss2_Pred'] = np.where(np.isnan(df[f'mmiss2_Pred'].values), df['mmiss2rec_brf'].values, df[f'mmiss2_Pred'].values)    
    df[f'BP_pick_pred']= np.where(cond2, np.ones_like(df[f'b0_P_Estm'].values), np.zeros_like(df[f'b0_P_Estm'].values)) 
    df[f'q2_pick_linreg']= np.where(np.isnan(df[f'q2_Pred'].values),np.ones_like(df['q2rec_quad'].values),                               np.zeros_like(df['q2rec_quad'].values)) 
    df[f'costhl_pick_linreg']=np.where(np.isnan(df[f'costhl_Pred'].values),np.ones_like(df['costhetalrec_quad'].values),
                                   np.zeros_like(df['costhetalrec_quad'].values)) 
    return df
'''
def get_fitvars(df, p3_Bfd, mB):
    """
    Build predicted fit variables using quadratic B-momentum solution and return_phasespace().
    Args:
        df: dataframe with input branches
        p3_Bfd: 3-momentum direction (unit vector) of B flight direction
        mB: B-meson mass
    """

    # D* and muon 4-vectors from reco branches
    Pdst_lab = atfk.lorentz_vector(
        atfk.vector(df['dst_PX'], df['dst_PY'], df['dst_PZ']), df['dst_PE']
    )
    Pl_lab = atfk.lorentz_vector(
        atfk.vector(df['mu_PX'], df['mu_PY'], df['mu_PZ']), df['mu_PE']
    )

    # D and pi 4-vectors
    PD_lab = atfk.lorentz_vector(
        atfk.vector(df['d0_PX'], df['d0_PY'], df['d0_PZ']), df['d0_PE']
    )
    Ppi_lab = atfk.lorentz_vector(
        atfk.vector(df['spi_PX'], df['spi_PY'], df['spi_PZ']), df['spi_PE']
    )


    # Quadratic solutions for B momentum magnitude
    df['b0_P_plus']  = return_quadsolution(
        p3_Bfd=p3_Bfd, m2_B=mB**2, p4_Dst=Pdst_lab, p4_mu=Pl_lab, solution='plus'
    ).numpy()
    df['b0_P_minus'] = return_quadsolution(
        p3_Bfd=p3_Bfd, m2_B=mB**2, p4_Dst=Pdst_lab, p4_mu=Pl_lab, solution='minus'
    ).numpy()

    # Choose solution closer to estimated value
    cond1 = np.less(
        np.abs(df['b0_P_Estm'].values - df['b0_P_plus'].values),
        np.abs(df['b0_P_Estm'].values - df['b0_P_minus'].values)
    )
    df['b0_P_Pred'] = np.where(cond1, df['b0_P_plus'].values, df['b0_P_minus'].values)

    # If both equal, fall back to estimate
    cond2 = np.equal(
        np.abs(df['b0_P_Estm'].values - df['b0_P_plus'].values),
        np.abs(df['b0_P_Estm'].values - df['b0_P_minus'].values)
    )
    df['b0_P_Pred'] = np.where(cond2, df['b0_P_Estm'].values, df['b0_P_Pred'].values)

    # Build B 4-vector with chosen momentum
    Pb0_lab = get4mom(p3_Bfd, df['b0_P_Pred'], mB)

    
    # Compute phasespace observables
    df['q2_Pred'], df['costhetal_Pred'], df['costhetad_Pred'],df['chi_Pred'],_ = return_phasespace(Pb0_lab, Pdst_lab, Pl_lab, PD_lab, Ppi_lab,Missmass2=False, convert_to_numpy=True)

    # Fallbacks if NaN
    df['q2_Pred']     = np.where(np.isnan(df['q2_Pred'].values),     df['q2rec_quad'].values,       df['q2_Pred'].values)
    df['costhetal_Pred'] = np.where(np.isnan(df['costhetal_Pred'].values), df['costhetalrec_quad'].values, df['costhetal_Pred'].values)
    df['costhetad_Pred'] = np.where(np.isnan(df['costhetad_Pred'].values), df['costhetadrec_quad'].values, df['costhetad_Pred'].values)
    df['chi_Pred']    = np.where(np.isnan(df['chi_Pred'].values),    df['chirec_quad'].values,       df['chi_Pred'].values)
    #df['mmiss2_Pred'] = np.where(np.isnan(df['mmiss2_Pred'].values), df['mmiss2rec_quad'].values,   df['mmiss2_Pred'].values)

    return df

def get4mom(p3_Bfd, p_B, mB):
    p3_Bfd_unit = atfk.unit_vector(p3_Bfd)
    p3_B        = atfk.scalar(p_B) * p3_Bfd_unit
    e_B         = atfi.sqrt(atfk.p(p3_B)**2 + mB**2) 
    return atfk.lorentz_vector(p3_B, e_B)

'''def return_phasespace(Pb0_lab, Pdst_lab, PLepton_lab, Missmass2 = True, convert_to_numpy = True):
    PW_lab       = Pb0_lab - Pdst_lab
    Pb0_Wlab     = atfk.boost_to_rest(Pb0_lab, PW_lab)                                               
    PLepton_Wlab = atfk.boost_to_rest(PLepton_lab, PW_lab)                            
    q2           = atfk.mass(PW_lab)**2
    costhl       = atfk.scalar_product(atfk.unit_vector(atfk.spatial_components(PLepton_Wlab)), -1. * atfk.unit_vector(atfk.spatial_components(Pb0_Wlab))) 
    PLepton_Blab= atfk.boost_to_rest(PLepton_lab, Pb0_lab)                            
    El           = atfk.time_component(PLepton_Blab)
    if Missmass2:         
        Pnu_lab = Pb0_lab - Pdst_lab - PLepton_lab
        MM2     = atfk.time_component(Pnu_lab)**2 - atfk.p(Pnu_lab)**2
        if convert_to_numpy: 
            return q2.numpy(), costhl.numpy(), El.numpy(), MM2.numpy()
        else:
            return q2, costhl, El, MM2
    else:
        if convert_to_numpy: 
            return q2.numpy(), costhl.numpy(), El.numpy()
        else:
            return q2, costhl, El
'''
def return_phasespace(Pb0_lab, Pdst_lab, PLepton_lab, PD_lab, Ppi_lab, Missmass2=True, convert_to_numpy=True):
    """
    Compute q2, cos(theta_l), cos(theta_d), chi, lepton energy, and missing mass squared.
    Args:
        Pb0_lab:     B0 4-vector in lab
        Pdst_lab:    D* 4-vector in lab
        PLepton_lab: lepton (mu) 4-vector in lab
        PD_lab:      D (daughter of D*) 4-vector in lab
        Ppi_lab:     pi (soft pion from D*->Dpi) 4-vector in lab
    """
    # -- q2 and cos(theta_l) --
    PW_lab       = Pb0_lab - Pdst_lab
    Pb0_Wlab     = atfk.boost_to_rest(Pb0_lab, PW_lab)
    PLepton_Wlab = atfk.boost_to_rest(PLepton_lab, PW_lab)
    q2           = atfk.mass(PW_lab)**2
    costhl       = atfk.scalar_product(
        atfk.unit_vector(atfk.spatial_components(PLepton_Wlab)),
        -1. * atfk.unit_vector(atfk.spatial_components(Pb0_Wlab))
    )
    PLepton_Blab = atfk.boost_to_rest(PLepton_lab, Pb0_lab)
    El           = atfk.time_component(PLepton_Blab)

    # -- cos(theta_d) --
    Pdst_Brest = atfk.boost_to_rest(Pdst_lab, Pb0_lab)
    PD_Brest   = atfk.boost_to_rest(PD_lab, Pb0_lab)
    Pb0_Brest  = atfk.boost_to_rest(Pb0_lab, Pb0_lab)

    PD_Dstrest  = atfk.boost_to_rest(PD_Brest, Pdst_Brest)
    Pb0_Dstrest = atfk.boost_to_rest(Pb0_Brest, Pdst_Brest)

    costhd = atfk.scalar_product(
        atfk.unit_vector(atfk.spatial_components(PD_Dstrest)),
        atfk.unit_vector(atfk.spatial_components(Pb0_Dstrest))
    )

    # -- chi angle --
    # neutrino momentum = Pb0 - Pdst - mu
    Pnu_lab = Pb0_lab - Pdst_lab - PLepton_lab

    # boost everything to B rest frame
    Pdst_Brest = atfk.boost_to_rest(Pdst_lab, Pb0_lab)
    PW_Brest   = atfk.boost_to_rest(PW_lab, Pb0_lab)
    Pnu_Brest  = atfk.boost_to_rest(Pnu_lab, Pb0_lab)
    Pmu_Brest  = atfk.boost_to_rest(PLepton_lab, Pb0_lab)
    Ppi_Brest  = atfk.boost_to_rest(Ppi_lab, Pb0_lab)
    PD_Brest   = atfk.boost_to_rest(PD_lab, Pb0_lab)

    # plane normals
    n_Dpi = atfk.cross_product(atfk.spatial_components(PD_Brest),
                       atfk.spatial_components(Ppi_Brest))
    n_munu = atfk.cross_product(atfk.spatial_components(Pmu_Brest),
                        atfk.spatial_components(Pnu_Brest))

    # basis for atan2
    coschi = atfk.scalar_product(atfk.unit_vector(n_munu),
                                 atfk.unit_vector(n_Dpi))
    sinchi = atfk.scalar_product(
        atfk.unit_vector(atfk.cross_product(n_munu, n_Dpi)),
        atfk.unit_vector(atfk.spatial_components(Pdst_Brest))
    )

    chi = atfi.atan2(sinchi, coschi)

    # -- missing mass squared --
    if Missmass2:
        MM2 = atfk.time_component(Pnu_lab)**2 - atfk.p(Pnu_lab)**2
        if convert_to_numpy:
            return (q2.numpy(), costhl.numpy(), costhd.numpy(),
                    chi.numpy(), El.numpy(), MM2.numpy())
        else:
            return q2, costhl, costhd, chi, El, MM2
    else:
        if convert_to_numpy:
            return (q2.numpy(), costhl.numpy(), costhd.numpy(),
                    chi.numpy(), El.numpy())
        else:
            return q2, costhl, costhd, chi, El

def return_quadsolution(p3_Bfd, m2_B, p4_Dst, p4_mu, solution = 'plus'):
    E_Dstmu_sq  = atfk.time_component(p4_Dst + p4_mu)**2
    m2_Dstmu    = atfk.mass(p4_Dst + p4_mu)**2
    p3_DstMu    = atfk.spatial_components(p4_Dst + p4_mu)
    cos_th     = atfk.scalar_product(p3_DstMu, p3_Bfd)/(atfk.p(p3_DstMu) * atfk.p(p3_Bfd))
    pDstmu_parl_sq = (atfk.p(p3_DstMu) * cos_th)**2
    pDstmu_perp_sq = atfk.p(p3_DstMu)**2 - pDstmu_parl_sq
    a   = ((m2_B - m2_Dstmu - 2. * pDstmu_perp_sq) * atfi.sqrt(pDstmu_parl_sq))/(2.*(pDstmu_parl_sq - E_Dstmu_sq))
    r_1 = ((m2_B - m2_Dstmu - 2. * pDstmu_perp_sq)**2 * E_Dstmu_sq)/(4.*(pDstmu_parl_sq - E_Dstmu_sq)**2) #B^2
    r_2 = (E_Dstmu_sq * pDstmu_perp_sq)/(pDstmu_parl_sq - E_Dstmu_sq) #-4AC
    r   = atfi.where(atfi.less(r_1, -r_2), atfi.zeros(r_1), r_1 + r_2) #If B^2 < - 4AC then zero (unique sol.) 
    pmiss_parl = None
    if solution == 'plus':
        pmiss_parl     = -a + atfi.sqrt(r)
    else:
        pmiss_parl     = -a - atfi.sqrt(r)

    p_B = atfi.sqrt(pDstmu_parl_sq) + pmiss_parl
    return p_B

def return_boostapprox(p3_Bfd, Pvis_lab, mB):
    p_B       = mB/atfk.mass(Pvis_lab) * atfk.z_component(atfk.spatial_components(Pvis_lab)) * 1./atfk.z_component(atfk.unit_vector(p3_Bfd))
    return p_B

def return_spherical(dfPX, dfPY, dfPZ):
    momvec = atfk.vector(dfPX, dfPY, dfPZ)
    mom = atfk.p(momvec)
    theta, phi = SphericalAngles(momvec)
    return mom, theta, phi

def Plot(df, var, suffix = None, res_limits = None, res_limits_rel = None, lin_limits = None, Dir = 'plots_tau', log = False, channel = 'mu', recono = False):
    if suffix: print(suffix)
    if channel == 'tau':
        lin_Reco = df[var+'mu_Reco'].values 
    else:
        lin_Reco = df[var+'_Reco'].values
        
        ##introduce q2rec_quad (our nominal resolution w/o training) to cross_product check the improvement
        lin_our_Reco = df['q2rec_quad'].values

    lin_TRUE = df[var+'_TRUE'].values 
    if suffix:
        lin_Pred = df[var+'_Pred'+suffix].values
    else:
        lin_Pred = df[var+'_Pred'].values
    
    if lin_limits: 
        cond = ((lin_Reco >= lin_limits[0]) & (lin_Reco <= lin_limits[1]) & (lin_Pred >= lin_limits[0]) & (lin_Pred <= lin_limits[1]) & (lin_TRUE >= lin_limits[0]) & (lin_TRUE <= lin_limits[1]))
        lin_Reco = lin_Reco[cond]
        lin_Pred = lin_Pred[cond]
        lin_TRUE = lin_TRUE[cond]
    
    res_reco = (lin_TRUE - lin_Reco)
    res_pred = (lin_TRUE - lin_Pred)
    ##introduce q2rec_quad (our nominal resolution w/o training) to cross_product check the improvement
    res_our_reco = lin_our_Reco - lin_TRUE
        
    if res_limits: 
        res_reco = res_reco[(res_reco >= res_limits[0]) & (res_reco <= res_limits[1])] 
        res_our_reco = res_our_reco[(res_our_reco >= res_limits[0]) & (res_our_reco <= res_limits[1])] 
        res_pred = res_pred[(res_pred >= res_limits[0]) & (res_pred <= res_limits[1])] 
       
    reco_strng = '$\mu={:0.2f}$, $\sigma={:0.2f}$'.format(np.mean(res_reco), np.std(res_reco))
    pred_strng = '$\mu={:0.2f}$, $\sigma={:0.2f}$'.format(np.mean(res_pred), np.std(res_pred))
    #
    our_reco_strng = '$\mu={:0.2f}$, $\sigma={:0.2f}$'.format(np.mean(res_our_reco), np.std(res_pred))

    if channel == 'tau':
        impv_strng = 'Improv in $\sigma$ wrt boost approx by {:0.2f}%'.format(100. * (1. - np.std(res_pred)/np.std(res_reco)))
    else:
        impv_strng = 'Resolution improvement by {:0.2f}%'.format(100. * (1. - np.std(res_pred)/np.std(res_reco)))
        #
        our_impv_strng = 'Resolution improvement by {:0.2f}%'.format(100. * (1. - np.std(res_pred)/np.std(res_our_reco)))

    lin_TRUE = lin_TRUE[np.nonzero(lin_TRUE)]
    lin_Pred = lin_Pred[np.nonzero(lin_TRUE)]
    lin_Reco = lin_Reco[np.nonzero(lin_TRUE)]
    #
    lin_our_Reco = lin_our_Reco[np.nonzero(lin_TRUE)]
    res_reco_rel = (lin_Reco - lin_TRUE)/lin_TRUE
    res_pred_rel = (lin_Pred - lin_TRUE)/lin_TRUE
    #
    res_our_reco_rel = (lin_our_Reco - lin_TRUE)/lin_TRUE
    numcands_before = res_reco_rel.shape[0]
    print('Reco Rel after', res_reco_rel.shape)
    
    if res_limits_rel: 
        res_reco_rel = res_reco_rel[(res_reco_rel >= res_limits_rel[0]) & (res_reco_rel <= res_limits_rel[1])] 
        res_pred_rel = res_pred_rel[(res_pred_rel >= res_limits_rel[0]) & (res_pred_rel <= res_limits_rel[1])]
        #
        res_our_reco_rel = res_our_reco_rel[(res_our_reco_rel >= res_limits_rel[0]) & (res_our_reco_rel <= res_limits_rel[1])] 
       
    numcands_after_reco = res_reco_rel.shape[0]
    numcands_after_pred = res_pred_rel.shape[0]
    res_reco_rel *= 100.
    res_pred_rel *= 100.
    #
    res_our_reco_rel *= 100.
    reco_mean_rel = np.mean(np.abs(res_reco_rel)); reco_std_rel  = np.std(np.abs(res_reco_rel))
    pred_mean_rel = np.mean(np.abs(res_pred_rel)); pred_std_rel  = np.std(np.abs(res_pred_rel))
    #
    our_reco_mean_rel = np.mean(np.abs(res_our_reco_rel)); our_reco_std_rel  = np.std(np.abs(res_our_reco_rel))
    reco_rel_strng = '$\mu={:0.2f}$, $\sigma={:0.2f}$ (#cands:{:d}/{:d})'.format(reco_mean_rel, reco_std_rel, numcands_after_reco, numcands_before)
    pred_rel_strng = '$\mu={:0.2f}$  $\sigma={:0.2f}$ (#cands:{:d}/{:d})'.format(pred_mean_rel, pred_std_rel, numcands_after_pred, numcands_before)
    #
    our_reco_rel_strng = '$\mu={:0.2f}$, $\sigma={:0.2f}$ (#cands:{:d}/{:d})'.format(our_reco_mean_rel, our_reco_std_rel, numcands_after_reco, numcands_before)
    if channel == 'tau':
        impv_rel_strng_mu  = 'Improv in $\mu$ wrt boost approx: {:0.2f}%'.format(100. * (1. - pred_mean_rel/reco_mean_rel))
        impv_rel_strng_sig = 'Improv in $\sigma$ wrt boost approx: {:0.2f}%'.format(100. * (1. - pred_std_rel/reco_std_rel))
    else:
        impv_rel_strng_mu  = 'Improv in $\mu$ wrt LSR: {:0.2f}%'.format(100. * (1. - pred_mean_rel/reco_mean_rel))
        impv_rel_strng_sig = 'Improv in $\sigma$ wrt LSR: {:0.2f}%'.format(100. * (1. - pred_std_rel/reco_std_rel))
        #
        our_impv_rel_strng_mu  = 'Improv in $\mu$ wrt LSR: {:0.2f}%'.format(100. * (1. - pred_mean_rel/our_reco_mean_rel))
        our_impv_rel_strng_sig = 'Improv in $\sigma$ wrt LSR: {:0.2f}%'.format(100. * (1. - pred_std_rel/our_reco_std_rel))

    fig, ax = plt.subplots()
    p = patches.Rectangle((left, bottom), width, height, fill=False, transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    if recono:
        ax.hist(res_pred, bins = 100, color='red'  , alpha=0.6, log=log, histtype = 'step', label = 'Pred')
    else:
        ax.hist(res_reco, bins = 100, color='green', alpha=0.6, log=log, histtype = 'step', label = 'Reconstructed')
        ax.hist(res_pred, bins = 100, color='red'  , alpha=0.6, log=log, histtype = 'step', label = 'Predicted')
        #
        #ax.hist(res_our_reco, bins = 100, color='blue'  , alpha=0.6, log=log, histtype = 'step', label = 'our reconstruction')
        ax.text(right, 0.7*(bottom+top), impv_strng, horizontalalignment='center', verticalalignment  ='center',fontsize = 11,transform=ax.transAxes, color = 'black')
        #ax.text(right, 0.7*(bottom+top), our_impv_strng, horizontalalignment='center', verticalalignment  ='center',fontsize = 11,transform=ax.transAxes, color = 'black')
    ax.legend(loc='best')
    ax.set_xlabel('$\Delta '+var+'$')
    if log:
        ax.set_ylabel(r'Log (Candidates per bin)')
        savename = Dir+'/'+var+'Resolution_log.pdf'
        if suffix: savename = Dir+'/'+var+'Resolution_log_'+suffix+'.pdf'
    else:
        ax.set_ylabel(r'Candidates per bin')
        savename = Dir+'/'+var+'Resolution_linear.pdf'
        if suffix: savename = Dir+'/'+var+'Resolution_linear_'+suffix+'.pdf'
    fig.tight_layout()
    fig.savefig(savename)
    del fig,ax

    fig, ax = plt.subplots()
    p = patches.Rectangle((left, bottom), width, height, fill=False, transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    if recono:
        ax.hist(res_pred_rel, bins = 100, color='red'  , alpha=0.6, log=log, histtype = 'step', label = 'Pred')
    else:
        ax.hist(res_reco_rel, bins = 100, color='green', alpha=0.6, log=log, histtype = 'step', label = 'Reconstructed')
        ax.hist(res_pred_rel, bins = 100, color='red'  , alpha=0.6, log=log, histtype = 'step', label = 'Predicted')
        #
        #ax.hist(res_our_reco_rel, bins = 100, color='blue', alpha=0.6, log=log, histtype = 'step', label = 'our reconstruction')
        ax.text(0.7*(left+right),0.5*(bottom+top),reco_rel_strng,horizontalalignment='center', verticalalignment  ='center',fontsize = 12,transform=ax.transAxes,color='green', alpha=0.6)
        ax.text(0.7*(left+right),0.42*(bottom+top),pred_rel_strng,horizontalalignment='center', verticalalignment  ='center',fontsize = 12, transform=ax.transAxes,color='red',alpha=0.6)
        #
        #ax.text(0.7*(left+right),0.34*(bottom+top),our_reco_rel_strng,horizontalalignment='center', verticalalignment  ='center',fontsize = 12,transform=ax.transAxes,color='blue', alpha=0.6)
    ax.legend(loc='best')
    ax.set_xlabel('$\Delta$ '+var+'/'+var+'$_{True}$ (%)')
    if log:
        ax.set_ylabel(r'Log (Candidates per bin)')
        savename = Dir+'/'+var+'RelResolution_log.pdf'
        if suffix: savename = Dir+'/'+var+'RelResolution_log_'+suffix+'.pdf'
    else:
        ax.set_ylabel(r'Candidates per bin')
        savename = Dir+'/'+var+'RelResolution_linear.pdf'
        if suffix: savename = Dir+'/'+var+'RelResolution_linear_'+suffix+'.pdf'
    fig.tight_layout()
    fig.savefig(savename)
    del fig,ax

    fig, ax = plt.subplots()
    ax.hist(lin_Reco, bins = 100, facecolor='green', alpha=0.6, density = True, log = log, label = 'Reco')
    ax.hist(lin_Pred, bins = 100, facecolor='red'  , alpha=0.6, density = True, log = log, label = 'Pred')
    ax.hist(lin_TRUE, bins = 100, facecolor='blue' , alpha=0.6, density = True, log = log, label = 'True')
    ax.legend(loc='best')
    ax.set_xlabel(var)
    ax.set_ylabel(r'Cands per bin')
    if log:
        ax.set_ylabel(r'Log (Cands per bin)')
        savename = Dir+'/'+var+'_log_'+suffix+'.pdf'
        if suffix: savename = Dir+'/'+var+'_log_'+suffix+'.pdf'
    else:
        ax.set_ylabel(r'Cands per bin')
        savename = Dir+'/'+var+'_linear_'+suffix+'.pdf'
        if suffix: savename = Dir+'/'+var+'_linear_'+suffix+'.pdf'
    fig.tight_layout()
    fig.savefig(savename)
    del fig,ax
