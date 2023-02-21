import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
from smp_utils import *




class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen=True, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29.):
        # should have separate lower ptcut for gen
        self.do_gen=do_gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]
                
        binning = util_binning()
        
        ptreco_axis = binning.ptreco_axis
        mreco_axis = binning.mreco_axis
        ptgen_axis = binning.ptgen_axis     
        mgen_axis = binning.mgen_axis

        dataset_axis = binning.dataset_axis
        lep_axis = binning.lep_axis
        n_axis = binning.n_axis
        mass_axis = binning.mass_axis
        zmass_axis = binning.zmass_axis
        pt_axis = binning.pt_axis
        frac_axis = binning.frac_axis
        dr_axis = binning.dr_axis
        dr_fine_axis = binning.dr_fine_axis
        dphi_axis = binning.dphi_axis    
        
        ### Plots of things during the selection process / for debugging with fine binning
        h_njet_gen = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        h_njet_reco = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        h_ptjet_gen_pre = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
        h_ptjet_reco_over_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_drjet_reco_gen = hist.Hist(dataset_axis, dr_fine_axis, storage="weight", label="Counts")
        h_mz_gen = hist.Hist(dataset_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco = hist.Hist(dataset_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco_over_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_dr_z_jet_gen = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dr_z_jet_reco = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dphi_z_jet_gen = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")
        h_dphi_z_jet_reco = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_reco = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_dr_gen_subjet = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        
        
        ### Plots to be unfolded
        h_ptjet_mjet_u_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, storage="weight", label="Counts")
        h_ptjet_mjet_g_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, storage="weight", label="Counts")
        ### Plots for comparison
        h_ptjet_mjet_u_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, storage="weight", label="Counts")        
        h_ptjet_mjet_g_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, storage="weight", label="Counts")
        
        
        ### Plots to get JMR and JMS in MC
        h_m_u_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")
        h_m_g_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")
        
        ### Plots for the analysis in the proper binning
        h_response_matrix_u = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, 
                                        storage="weight", label="Counts")
        h_response_matrix_g = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, 
                                        storage="weight", label="Counts")
        
        cutflow = {}
        
        self.hists = {
            "njet_gen":h_njet_gen,
            "njet_reco":h_njet_reco,
            "ptjet_gen_pre":h_ptjet_gen_pre, 
            "ptjet_mjet_u_gen":h_ptjet_mjet_u_gen, 
            "ptjet_mjet_u_reco":h_ptjet_mjet_u_reco, 
            "ptjet_mjet_g_gen":h_ptjet_mjet_g_gen, 
            "ptjet_mjet_g_reco":h_ptjet_mjet_g_reco, 
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
        
        
        
        #####################################
        ### Initialize selection
        #####################################
        selection = PackedSelection()
        
        
         
        #####################################
        ### Trigger selection for data
        #####################################       
        if not self.do_gen:
            if "UL2016" in dataset: 
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele27_WPTight_Gsf | events.HLT.Photon175
            elif "UL2017" in dataset:
                trigsel = events.HLT.IsoMu27 | events.HLT.Ele35_WPTight_Gsf | events.HLT.Photon200
            elif "UL2018" in dataset:
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200
            elif "Test" in dataset: 
                trigsel = events.HLT.IsoMu24 | events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200
            else:
                raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
            selection.add("trigsel", trigsel)    
        
        #####################################
        ### Remove events with very large gen weights (>2 sigma)
        #####################################
        if self.do_gen:
            if dataset not in self.means_stddevs : 
                average = np.average( events["LHEWeight"].originalXWGTUP )
                stddev = np.std( events["LHEWeight"].originalXWGTUP )
                self.means_stddevs[dataset] = (average, stddev)            
            average,stddev = self.means_stddevs[dataset]
            vals = (events["LHEWeight"].originalXWGTUP - average ) / stddev
            self.hists["cutflow"][dataset]["all events"] += len(events)
            events = events[ np.abs(vals) < 2 ]
            self.hists["cutflow"][dataset]["weights cut"] += len(events)

            #####################################
            ### Initialize event weight to gen weight
            #####################################
            weights = events["LHEWeight"].originalXWGTUP
        else:
            weights = np.full( len( events ), 1.0 )
        
        


        
        #####################################
        #####################################
        #####################################
        ### Gen selection
        #####################################
        #####################################
        #####################################
        if self.do_gen:
            #####################################
            ### Events with at least one gen jet
            #####################################
            selection.add("oneGenJet", 
                  ak.sum( (events.GenJetAK8.pt > 120.) & (np.abs(events.GenJetAK8.eta) < 2.5), axis=1 ) >= 1
            )

            #####################################
            ### Make gen-level Z
            #####################################
            z_gen = get_z_gen_selection(events, selection, self.lepptcuts[0], self.lepptcuts[1] )
            z_gen_cuts = ak.where( selection.all("twoGen_leptons") & ~ak.is_none(z_gen),  z_gen.pt > 90., False )
            selection.add("z_gen_pt", z_gen_cuts)

            #####################################
            ### Get Gen Jet
            #####################################
            gen_jet, dphi_gen_sel = find_opposite( z_gen, events.GenJetAK8 )
            zgen_ngenjet_sel = selection.require(z_gen_pt=True,oneGenJet=True)

            #####################################
            ### Gen event topology selection
            #####################################        
            selection.add("z_gen_jet_dphi", dphi_gen_sel)
            z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
            z_pt_asym_gen_sel = z_pt_asym_gen < 0.3
            selection.add("z_pt_asym_gen_sel", z_pt_asym_gen_sel)

            #####################################
            ### Make gen plots with a Z candidate and at least one jet
            #####################################

            self.hists["mz_gen"].fill(dataset=dataset,
                                      mass=z_gen[zgen_ngenjet_sel].mass,
                                      weight=weights[zgen_ngenjet_sel])
            self.hists["njet_gen"].fill(dataset=dataset,
                                        n=ak.num(events[zgen_ngenjet_sel].GenJetAK8),
                                        weight = weights[zgen_ngenjet_sel] )
            self.hists["dphi_z_jet_gen"].fill(dataset=dataset, dphi=z_gen[zgen_ngenjet_sel].delta_phi(gen_jet[zgen_ngenjet_sel]), 
                                       weight=weights[zgen_ngenjet_sel])
            self.hists["dr_z_jet_gen"].fill(dataset=dataset, dr=z_gen[zgen_ngenjet_sel].delta_r(gen_jet[zgen_ngenjet_sel]), 
                                       weight=weights[zgen_ngenjet_sel])
            self.hists["ptasym_z_jet_gen"].fill(dataset=dataset, frac=z_pt_asym_gen[zgen_ngenjet_sel], 
                                       weight=weights[zgen_ngenjet_sel])

            #####################################
            ### Get gen subjets 
            #####################################
            gensubjets = events.SubGenJetAK8
            groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)

            #####################################
            ### Convenience selection that has all gen cuts
            #####################################
            all_gen_cuts = selection.all("z_gen_pt", "z_gen_jet_dphi", "z_pt_asym_gen_sel")
            selection.add("all_gen_cuts", all_gen_cuts)

            #####################################
            ### Plots for gen jets and subjets
            #####################################
            self.hists["ptjet_gen_pre"].fill(dataset=dataset, 
                                         pt=gen_jet[all_gen_cuts].pt, 
                                         weight=weights[all_gen_cuts])
            self.hists["dr_gen_subjet"].fill(dataset=dataset,
                                             dr=groomed_gen_jet[all_gen_cuts].delta_r(gen_jet[all_gen_cuts]),
                                             weight=weights[all_gen_cuts])
                        
            
        #####################################
        ### Make reco-level Z
        #####################################
        z_reco = get_z_reco_selection(events, selection, self.lepptcuts[0], self.lepptcuts[1])
        z_reco_cuts = ak.where( selection.all("twoReco_leptons") & ~ak.is_none(z_reco), z_reco.pt > 90., False )
        selection.add("z_reco_pt", z_reco_cuts)


        #####################################
        ### Reco jet selection
        #####################################
        recojets = events.FatJet
        selection.add("oneRecoJet", 
             ak.sum( (events.FatJet.pt > 170.) & (np.abs(events.FatJet.eta) < 2.5), axis=1 ) >= 1
        )
        self.hists["njet_reco"].fill(dataset=dataset, n=ak.num(recojets,axis=1))
        
        # Find reco jet closest to the gen jet
        if self.do_gen:
            reco_jet,reco_jet_dr = find_closest_dr( gen_jet, recojets, verbose=False)
            jet_matching_cuts = np.logical_not( ak.is_none(reco_jet) ) & (reco_jet_dr < 0.2)
            selection.add("jet_genjet_matching_cuts", jet_matching_cuts)
        else :
            reco_jet, dphi_reco_sel = find_opposite( z_reco, events.FatJet )
            

        #####################################
        ### Convenience selection that has all gen cuts and reco preselection
        #####################################
        if self.do_gen:
            reco_preselection = selection.all("all_gen_cuts", "twoReco_leptons", "z_reco_pt", "oneRecoJet", "jet_genjet_matching_cuts")
        else:
            reco_preselection = selection.all("trigsel", "twoReco_leptons", "z_reco_pt", "oneRecoJet")
        selection.add("reco_preselection", reco_preselection)

        
        #####################################
        ### Reco event topology selection
        #####################################
        z_dphi_reco = z_reco.delta_phi( reco_jet )
        z_dphi_reco_sel = np.abs(z_dphi_reco) > np.pi * 0.5
        z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
        z_pt_asym_reco_sel = z_pt_asym_reco < 0.3
        selection.add("z_dphi_reco_sel", z_dphi_reco_sel)
        selection.add("z_pt_asym_reco_sel", z_pt_asym_reco_sel)
        
        
        #####################################
        ### Make reco plots with a Z candidate and at least one jet. 
        ### These use the event topology variables just defined and make a plot. 
        #####################################
        self.hists["mz_reco"].fill(dataset=dataset, mass=z_reco[reco_preselection].mass, 
                                   weight=weights[reco_preselection])
        if self.do_gen:
            self.hists["mz_reco_over_gen"].fill(dataset=dataset, 
                                                frac=z_reco[reco_preselection].mass / z_gen[reco_preselection].mass, 
                                                weight=weights[reco_preselection] )
        self.hists["dr_z_jet_reco"].fill( dataset=dataset,dr=z_reco[reco_preselection].delta_r(reco_jet[reco_preselection]),
                                         weight=weights[reco_preselection] )
        self.hists["dphi_z_jet_reco"].fill(dataset=dataset, dphi=z_dphi_reco[reco_preselection], 
                                           weight=weights[reco_preselection])
        self.hists["ptasym_z_jet_reco"].fill(dataset=dataset, frac=z_pt_asym_reco[reco_preselection],
                                             weight=weights[reco_preselection])

        #####################################
        ### Convenience selection that has all cuts
        #####################################
        if self.do_gen:
            final_selection =  selection.all("all_gen_cuts", "z_reco_pt", "oneGenJet", 
                                             "jet_genjet_matching_cuts", "z_dphi_reco_sel", "z_pt_asym_reco_sel" )
        else:
            final_selection =  selection.all("z_reco_pt", "oneRecoJet", "z_dphi_reco_sel", "z_pt_asym_reco_sel" )

            
        selection.add("final_selection", final_selection )
        
        #####################################
        ### Make final selection plots here
        #####################################
        
        # For convenience, finally reduce the size of the arrays at the end
        weights = weights[final_selection]
        z_reco = z_reco[final_selection]
        reco_jet = reco_jet[final_selection]
        self.hists["ptjet_mjet_u_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.mass, weight=weights )
        self.hists["ptjet_mjet_g_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.msoftdrop, weight=weights )
        
        if self.do_gen:
            z_gen = z_gen[final_selection]
            gen_jet = gen_jet[final_selection]
            groomed_gen_jet = groomed_gen_jet[final_selection]


            self.hists["ptjet_mjet_u_gen"].fill( dataset=dataset, ptgen=gen_jet.pt, mgen=gen_jet.mass, weight=weights )
            self.hists["ptjet_mjet_g_gen"].fill( dataset=dataset, ptgen=gen_jet.pt, mgen=groomed_gen_jet.mass, weight=weights )
            
            self.hists["drjet_reco_gen"].fill(dataset=dataset, dr=reco_jet.delta_r(gen_jet), weight=weights)
            
        
            self.hists["ptjet_reco_over_gen"].fill(dataset=dataset, frac=reco_jet.pt/gen_jet.pt, weight=weights)
            self.hists["m_u_jet_reco_over_gen"].fill(dataset=dataset, 
                                                     ptgen=gen_jet.pt, mgen = gen_jet.mass, 
                                                     frac=reco_jet.mass/gen_jet.mass, weight=weights)
            self.hists["m_g_jet_reco_over_gen"].fill(dataset=dataset, 
                                                     ptgen=gen_jet.pt, mgen=groomed_gen_jet.mass,
                                                     frac=reco_jet.msoftdrop/groomed_gen_jet.mass, weight=weights)


            self.hists["response_matrix_u"].fill( dataset=dataset, 
                                               ptreco=reco_jet.pt, ptgen=gen_jet.pt,
                                               mreco=reco_jet.mass, mgen=gen_jet.mass )
            self.hists["response_matrix_g"].fill( dataset=dataset, 
                                               ptreco=reco_jet.pt, ptgen=gen_jet.pt,
                                               mreco=reco_jet.msoftdrop, mgen=groomed_gen_jet.mass )

        
        for name in selection.names:
            self.hists["cutflow"][dataset][name] = selection.all(name).sum()
        
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator
    
    
    
    