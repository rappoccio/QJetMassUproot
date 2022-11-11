import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict

def find_closest_dr( a, coll , verbose = False):
    """
    Find the objects within coll that are closest to a. 
    Return it and the delta R between it and a.
    """
    combs = ak.cartesian( (a, coll), axis=1 )
    dr = combs['0'].delta_r(combs['1'])
    dr_min = ak.singletons( ak.argmin( dr, axis=1 ) )
    sel = combs[dr_min]['1']
    return ak.firsts(sel),ak.firsts(dr[dr_min])
    
    

def get_groomed_jet( jet, subjets , verbose = False):
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.4]
    total = combs['1'].sum(axis=1)
    return total

    
def find_opposite( a, coll, dphimin = 1, verbose=False ):
    """
    Find the highest-pt object in coll in the hemisphere opposite a,
    defined by delta phi being > dphimin (default 1). 
    Return it and the selection boolean vector. 
    """
    combs = ak.cartesian( (a, coll), axis=1 )
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    sel = dphi > dphimin
    coll_opposite = combs[ sel ]
    return ak.firsts( coll_opposite['1'] ), sel




class QJetMassProcessor(processor.ProcessorABC):
    def __init__(self, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29.):
        # should have separate lower ptcut for gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]
        
        ptreco_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")        
        mreco_axis = hist.axis.Variable([0,5,10,20,40,60,80,100,150,200,250,300,350,1000], name="mreco", label=r"m_{RECO} (GeV)")
        ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")                
        mgen_axis = hist.axis.Variable( [0,2.5,5,7.5,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350,1000], name="mgen", label=r"Mass [GeV]")

        
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        lep_axis = hist.axis.StrCategory(["ee", "mm"], name="lep")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        mass_axis = hist.axis.Regular(100, 0, 1000, name="mass", label=r"$m$ [GeV]")
        zmass_axis = hist.axis.Regular(100, 80, 100, name="mass", label=r"$m$ [GeV]")
        pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")                
        frac_axis = hist.axis.Regular(150, 0, 2.0, name="frac", label=r"Fraction")                
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dr_fine_axis = hist.axis.Regular(150, 0, 1.5, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")       
        
        ### Plots of things during the selection process / for debugging with fine binning
        h_njet_gen = hist.Hist(dataset_axis, lep_axis, n_axis, storage="weight", label="Counts")
        h_njet_reco = hist.Hist(dataset_axis, lep_axis, n_axis, storage="weight", label="Counts")
        h_ptjet_gen = hist.Hist(dataset_axis, lep_axis, pt_axis, storage="weight", label="Counts")
        h_ptjet_reco = hist.Hist(dataset_axis, lep_axis, pt_axis, storage="weight", label="Counts")
        h_ptjet_reco_over_gen = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        h_drjet_reco_gen = hist.Hist(dataset_axis, lep_axis, dr_fine_axis, storage="weight", label="Counts")
        h_mz_gen = hist.Hist(dataset_axis, lep_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco = hist.Hist(dataset_axis, lep_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco_over_gen = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        h_dr_z_jet_gen = hist.Hist(dataset_axis, lep_axis, dr_axis, storage="weight", label="Counts")
        h_dr_z_jet_reco = hist.Hist(dataset_axis, lep_axis, dr_axis, storage="weight", label="Counts")
        h_dphi_z_jet_gen = hist.Hist(dataset_axis, lep_axis, dphi_axis, storage="weight", label="Counts")
        h_dphi_z_jet_reco = hist.Hist(dataset_axis, lep_axis, dphi_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_gen = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_reco = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        h_dr_gen_subjet = hist.Hist(dataset_axis, lep_axis, dr_axis, storage="weight", label="Counts")
        h_m_u_jet_reco_over_gen = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        h_m_g_jet_reco_over_gen = hist.Hist(dataset_axis, lep_axis, frac_axis, storage="weight", label="Counts")
        
        ### Plots for the analysis in the proper binning
        h_response_matrix_u = hist.Hist(dataset_axis, lep_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, 
                                        storage="weight", label="Counts")
        h_response_matrix_g = hist.Hist(dataset_axis, lep_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, 
                                        storage="weight", label="Counts")
        
        cutflow = {}
        
        self.hists = {
            "njet_gen":h_njet_gen,
            "njet_reco":h_njet_reco,
            "ptjet_gen":h_ptjet_gen, 
            "ptjet_reco":h_ptjet_reco, 
            "ptjet_reco_over_gen":h_ptjet_reco_over_gen,
            "drjet_reco_gen":h_drjet_reco_gen,
            "mz_gen":h_mz_gen,
            "mz_reco":h_mz_reco,
            "mz_reco_over_gen":h_mz_reco_over_gen,
            "dr_z_jet_gen":h_dr_z_jet_gen,
            "dr_z_jet_reco":h_dr_z_jet_reco,            
            "dphi_z_jet_gen":h_dphi_z_jet_gen,
            "dphi_z_jet_reco":h_dphi_z_jet_reco,
            "ptasym_z_jet_gen":h_ptasym_z_jet_gen,
            "ptasym_z_jet_reco":h_ptasym_z_jet_reco,
            "m_u_jet_reco_over_gen":h_m_u_jet_reco_over_gen,
            "m_g_jet_reco_over_gen":h_m_g_jet_reco_over_gen,
            "dr_gen_subjet":h_dr_gen_subjet,
            "response_matrix_u":h_response_matrix_u,
            "response_matrix_g":h_response_matrix_g,
            "cutflow":cutflow
        }
        
        ## This is for rejecting events with large weights
        self.means_stddevs = defaultdict()
    
    @property
    def accumulator(self):
        return self._histos

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        dataset = events.metadata['dataset']
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
        
        
        ## Remove events with very large weights (>2 sigma)
        if dataset not in self.means_stddevs : 
            average = np.average( events["LHEWeight"].originalXWGTUP )
            stddev = np.std( events["LHEWeight"].originalXWGTUP )
            self.means_stddevs[dataset] = (average, stddev)
            
        average,stddev = self.means_stddevs[dataset]
        vals = (events["LHEWeight"].originalXWGTUP - average ) / stddev
        
        self.hists["cutflow"][dataset]["all events"] += len(events)
        
        events = events[ np.abs(vals) < 2 ]
        self.hists["cutflow"][dataset]["weights cut"] += len(events)
        

            
        

        genelectrons = events.GenDressedLepton[ np.abs(events.GenDressedLepton.pdgId) == 11]
        genmuons = events.GenDressedLepton[ np.abs(events.GenDressedLepton.pdgId) == 13]        
        genleptons = [genelectrons, genmuons]    
        recoleptons = [events.Electron, events.Muon]
        
        
        for ilep,lepstr in dict(zip( [0,1], ["ee", "mm"] )).items():
            
            weights = events["LHEWeight"].originalXWGTUP
            #####################################
            # Gen lepton and Z selection
            #####################################
            self.hists["cutflow"][dataset][lepstr + " total"] += len(genleptons[ilep])
            
            genlepsel = (genleptons[ilep].pt > self.lepptcuts[ilep]) & (np.abs(genleptons[ilep].eta) < 2.5)
            genleptons[ilep] = genleptons[ilep][ genlepsel ]            
            nlep=ak.num(genleptons[ilep])
            isDilepGen = (nlep >= 2)
            self.hists["cutflow"][dataset][lepstr + " nlep >=2"] += ak.count_nonzero(isDilepGen)
            weights = weights[isDilepGen]
            genleptons[ilep] = genleptons[ilep][isDilepGen]
            z_gen = genleptons[ilep][:,0] + genleptons[ilep][:,1]
            z_gen_ptsel = z_gen.pt > 90.
            z_gen = z_gen[ z_gen_ptsel ]
            self.hists["cutflow"][dataset][lepstr + " zpt > 90"] += ak.count_nonzero( z_gen_ptsel ) 
            weights = weights[ z_gen_ptsel ]
            
            self.hists["njet_gen"].fill(dataset=dataset,
                                        lep=lepstr, 
                                        n=ak.num(events[isDilepGen][z_gen_ptsel].GenJetAK8),
                                        weight = weights )
            n_gen_jet_sel = ak.num( events[isDilepGen][z_gen_ptsel].GenJetAK8,axis=1 ) > 0
            
            

            
            #####################################
            # Gen jet selection
            #####################################
            
            # Find highest pt jet that is at least 1 radian in phi from the Z
            z_gen = z_gen[n_gen_jet_sel]
            gen_jet, gen_jet_found_sel = find_opposite( z_gen, events[isDilepGen][z_gen_ptsel][n_gen_jet_sel].GenJetAK8)
            gen_jet_n = np.logical_not( ak.is_none(gen_jet) )
            gen_jet = gen_jet[gen_jet_n]
            z_gen = z_gen[gen_jet_n]
            weights = weights[gen_jet_n]
            self.hists["mz_gen"].fill(dataset=dataset,lep=lepstr, mass=z_gen.mass,weight=weights)
            self.hists["cutflow"][dataset][lepstr + " >=1 gen jet"] += ak.count_nonzero(gen_jet_n)
            
            gen_jet_ptsel = gen_jet.pt > 170.
            gen_jet_etasel = np.abs(gen_jet.eta) < 2.5
            z_dphi_gen = z_gen.delta_phi( gen_jet )
            z_dphi_gen_sel = np.abs(z_dphi_gen) > np.pi * 0.5
            z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
            z_pt_asym_gen_sel = z_pt_asym_gen < 0.3
            
            self.hists["cutflow"][dataset][lepstr + " gen jet pt cut"] += ak.count_nonzero(gen_jet_ptsel)
            self.hists["cutflow"][dataset][lepstr + " gen jet eta cut"] += ak.count_nonzero(gen_jet_ptsel & gen_jet_etasel)
            self.hists["cutflow"][dataset][lepstr + " gen jet dphi cut"] += ak.count_nonzero(gen_jet_ptsel & gen_jet_etasel & z_dphi_gen_sel)
            self.hists["cutflow"][dataset][lepstr + " gen jet asym cut"] += ak.count_nonzero(gen_jet_ptsel & gen_jet_etasel & z_dphi_gen_sel & z_pt_asym_gen_sel)

            self.hists["dphi_z_jet_gen"].fill(dataset=dataset,lep=lepstr, dphi=z_dphi_gen,weight=weights)
            self.hists["ptasym_z_jet_gen"].fill(dataset=dataset,lep=lepstr, frac=z_pt_asym_gen,weight=weights)
            
            gen_jet_sel = gen_jet_ptsel & gen_jet_etasel & z_dphi_gen_sel & z_pt_asym_gen_sel
            weights = weights[gen_jet_sel]
            gen_jet = gen_jet[gen_jet_sel]
            z_gen = z_gen[gen_jet_sel]
            self.hists["cutflow"][dataset][lepstr + " gen jet cuts"] += ak.count_nonzero(gen_jet_sel)
            self.hists["ptjet_gen"].fill(dataset=dataset,lep=lepstr, pt=gen_jet.pt, weight=weights)

            
            
            #####################################
            # Gen subjets selection
            #####################################

            gensubjets = events[isDilepGen][z_gen_ptsel][n_gen_jet_sel][gen_jet_n][gen_jet_sel].SubGenJetAK8
            groomed_gen_jet = get_groomed_jet(gen_jet, gensubjets, False)
            groomedgensel = ~ak.is_none(groomed_gen_jet)
            z_gen = z_gen[groomedgensel]
            gen_jet = gen_jet[groomedgensel]
            weights = weights[groomedgensel]
            groomed_gen_jet = groomed_gen_jet[groomedgensel]
            self.hists["cutflow"][dataset][lepstr + " groomed gen jet cuts "] += ak.count_nonzero(groomedgensel)
            self.hists["dr_gen_subjet"].fill(dataset=dataset,lep=lepstr, dr=groomed_gen_jet.delta_r(gen_jet), weight=weights)
                        
            #####################################
            # Reco lepton and Z selection
            #####################################
            self.hists["cutflow"][dataset][lepstr + " total"] += len(genleptons[ilep])
            recoleptons[ilep] = recoleptons[ilep][isDilepGen][z_gen_ptsel][n_gen_jet_sel][gen_jet_n][gen_jet_sel][groomedgensel]

            recolepsel = (recoleptons[ilep].pt > self.lepptcuts[ilep]) & (np.abs(recoleptons[ilep].eta) < 2.5)
            recoleptons[ilep] = recoleptons[ilep][ recolepsel ]            
            nlep=ak.num(recoleptons[ilep])
            isDilepReco = (nlep >= 2)
            gen_jet = gen_jet[isDilepReco]
            groomed_gen_jet = groomed_gen_jet[isDilepReco]
            z_gen = z_gen[isDilepReco]
            weights = weights[isDilepReco]
            self.hists["cutflow"][dataset][lepstr + " nlep >=2"] += ak.count_nonzero(isDilepReco)

            
            recoleptons[ilep] = recoleptons[ilep][isDilepReco]
            z_reco = recoleptons[ilep][:,0] + recoleptons[ilep][:,1]
            z_reco_ptsel = z_reco.pt > 90.
            z_reco = z_reco[ z_reco_ptsel ]
            gen_jet = gen_jet[ z_reco_ptsel ]
            groomed_gen_jet = groomed_gen_jet[z_reco_ptsel]
            z_gen = z_gen[ z_reco_ptsel ]
            weights = weights[z_reco_ptsel]
            self.hists["dr_z_jet_gen"].fill( dataset=dataset,lep=lepstr,dr=z_gen.delta_r(gen_jet), weight=weights )
            self.hists["cutflow"][dataset][lepstr + " zpt > 90"] += ak.count_nonzero(z_reco_ptsel)
            


            #####################################
            # Reco jet selection
            #####################################
            recojets = events[isDilepGen][z_gen_ptsel][n_gen_jet_sel][gen_jet_n][gen_jet_sel][isDilepReco][z_reco_ptsel].FatJet            
            self.hists["njet_reco"].fill(dataset=dataset,lep=lepstr, n=ak.num(recojets,axis=1))
            n_reco_jet_sel = ak.num( recojets,axis=1 ) > 0
            
            recojets = recojets[n_reco_jet_sel]
            gen_jet = gen_jet[n_reco_jet_sel]
            groomed_gen_jet = groomed_gen_jet[n_reco_jet_sel]
            z_gen = z_gen[n_reco_jet_sel]
            z_reco = z_reco[n_reco_jet_sel]
            weights = weights[n_reco_jet_sel]
            
            # Find reco jet closest to the gen jet
            reco_jet,reco_jet_dr = find_closest_dr( gen_jet, recojets, verbose=False)
            reco_jet_n = np.logical_not( ak.is_none(reco_jet) ) & (reco_jet_dr < 0.2)
            gen_jet = gen_jet[reco_jet_n]
            groomed_gen_jet = groomed_gen_jet[reco_jet_n]
            reco_jet = reco_jet[reco_jet_n]
            z_reco = z_reco[reco_jet_n]
            z_gen = z_gen[reco_jet_n]
            weights = weights[reco_jet_n]
            self.hists["mz_reco"].fill(dataset=dataset,lep=lepstr, mass=z_reco.mass, weight=weights)
            self.hists["mz_reco_over_gen"].fill(dataset=dataset,lep=lepstr, frac=z_reco.mass / z_gen.mass, weight=weights )
            self.hists["cutflow"][dataset][lepstr + " >=1 reco jet"] += ak.count_nonzero(reco_jet_n)
            
            reco_jet_ptsel = reco_jet.pt > 170.
            reco_jet_etasel = np.abs(reco_jet.eta) < 2.5
            z_dphi_reco = z_reco.delta_phi( reco_jet )
            z_dphi_reco_sel = np.abs(z_dphi_reco) > np.pi * 0.5
            z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
            z_pt_asym_reco_sel = z_pt_asym_reco < 0.3
            self.hists["dr_z_jet_reco"].fill( dataset=dataset,lep=lepstr,dr=z_reco.delta_r(reco_jet), weight=weights )
            self.hists["dphi_z_jet_reco"].fill(dataset=dataset,lep=lepstr, dphi=z_dphi_reco, weight=weights)
            self.hists["ptasym_z_jet_reco"].fill(dataset=dataset,lep=lepstr, frac=z_pt_asym_reco, weight=weights)
            reco_jet_sel = reco_jet_ptsel & reco_jet_etasel & z_dphi_reco_sel & z_pt_asym_reco_sel
            reco_jet = reco_jet[reco_jet_sel]
            z_reco = z_reco[reco_jet_sel]
            gen_jet = gen_jet[reco_jet_sel]
            groomed_gen_jet = groomed_gen_jet[reco_jet_sel]
            weights = weights[reco_jet_sel]

            
            self.hists["cutflow"][dataset][lepstr + " reco jet cuts"] += len(reco_jet_sel)
            self.hists["drjet_reco_gen"].fill(dataset=dataset,lep=lepstr, dr=reco_jet.delta_r(gen_jet), weight=weights)
            self.hists["ptjet_reco"].fill(dataset=dataset,lep=lepstr, pt=reco_jet.pt, weight=weights)
            self.hists["ptjet_reco_over_gen"].fill(dataset=dataset,lep=lepstr, frac=reco_jet.pt/gen_jet.pt, weight=weights)
            
            
            
            #####################################
            # Plots with full selection
            #####################################
            
            self.hists["m_u_jet_reco_over_gen"].fill(dataset=dataset,lep=lepstr, frac=reco_jet.mass/gen_jet.mass, weight=weights)
            self.hists["m_g_jet_reco_over_gen"].fill(dataset=dataset,lep=lepstr, frac=reco_jet.msoftdrop/groomed_gen_jet.mass, weight=weights)

            
            self.hists["response_matrix_u"].fill( dataset=dataset, lep=lepstr, 
                                               ptreco=reco_jet.pt, ptgen=gen_jet.pt,
                                               mreco=reco_jet.mass, mgen=gen_jet.mass )
            self.hists["response_matrix_g"].fill( dataset=dataset, lep=lepstr, 
                                               ptreco=reco_jet.pt, ptgen=gen_jet.pt,
                                               mreco=reco_jet.msoftdrop, mgen=groomed_gen_jet.mass )
            
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator
    
    
    
    