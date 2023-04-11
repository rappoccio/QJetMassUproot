#############################################################################
#### Author : Garvita Agarwal
#############################################################################


import time
from coffea import nanoevents, util
import hist
import coffea.processor as processor
import awkward as ak
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import glob as glob
import re
import itertools
import vector as vec
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoAODSchema
from coffea.lumi_tools import LumiMask
# for applying JECs
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
#from jmeCorrections import ApplyJetCorrections, corrected_polar_met
from collections import defaultdict
import correctionlib


def CorrectJetsRun2(IOV, Jets, JetsName="AK4PFchs", leptons_inJet=None):
    jer_tag=None
    if (IOV=='2018' or IOV == 'Test'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "RunA": "Summer19UL18_RunA_V5_DATA",
            "RunB": "Summer19UL18_RunB_V5_DATA",
            "RunC": "Summer19UL18_RunC_V5_DATA",
            "RunD": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "RunB": "Summer19UL17_RunB_V5_DATA",
            "RunC": "Summer19UL17_RunC_V5_DATA",
            "RunD": "Summer19UL17_RunD_V5_DATA",
            "RunE": "Summer19UL17_RunE_V5_DATA",
            "RunF": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV2_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V5_MC"
        jec_tag_data={
            "RunF": "Summer19UL16_RunFGH_V7_DATA",
            "RunG": "Summer19UL16_RunFGH_V7_DATA",
            "RunH": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V5_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "RunB_ver1": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunB_ver2": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunC": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunD": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunE": "Summer19UL16APV_RunEF_V7_DATA",
            "RunF": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        raise ValueError(f"Error: Unknown year \"{IOV}\".")
    
    extract = extractor()
    if (isMC):
        #For MC
        extract.add_weight_sets([
            '* * data/JEC/{0}/{0}_L1FastJet_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_L2Relative_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_L3Absolute_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_UncertaintySources_{1}.junc.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_Uncertainty_{1}.junc.txt'.format(jec_tag, JetsName),
        ])

        if jer_tag:
            extract.add_weight_sets([
            '* * data/JER/{0}/{0}_PtResolution_{1}.jr.txt'.format(jer_tag, JetsName),
            '* * data/JER/{0}/{0}_SF_{1}.jersf.txt'.format(jer_tag, JetsName)])
    else:       
        #For data, make sure we don't duplicate
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                extract.add_weight_sets([
                '* * data/JEC/{0}/{0}_L1FastJet_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L2Relative_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L3Absolute_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L2L3Residual_{1}.txt'.format(tag, JetsName),
                ])
                tags_done += [tag]
                
    extract.finalize()
    evaluator = extract.make_evaluator()
    
    if (isMC):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, JetsName),
            '{0}_L2Relative_{1}'.format(jec_tag, JetsName),
            '{0}_L3Absolute_{1}'.format(jec_tag, JetsName),
            '{0}_Uncertainty_{1}'.format(jec_tag, JetsName)]
        if do_factorized_jec_unc:
            for name in dir(evaluator):
               #factorized sources
               if '{0}_UncertaintySources_{1}'.format(jec_tag, JetsName) in name:
                    jec_names.append(name)
        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, JetsName),
                              '{0}_SF_{1}'.format(jer_tag, JetsName)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, JetsName),
                '{0}_L3Absolute_{1}'.format(tag, JetsName),
                '{0}_L2Relative_{1}'.format(tag, JetsName),
                '{0}_L2L3Residual_{1}'.format(tag, JetsName),]
    if isMC:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}
    
    
    
    
    ## (1) Uncorrecting Jets
    CleanedJets = Jets
    debug(self.debugMode, "Corrected Jets (Before Cleaning): ", CleanedJets[0].pt)
    CleanedJets["rho"] = ak.broadcast_arrays(df.fixedGridRhoFastjetAll, Jets.pt)[0]
    CleanedJets["pt_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.pt
    CleanedJets["mass_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.mass
    CleanedJets["pt"] = CleanedJets.pt_raw
    CleanedJets["mass"] = CleanedJets.mass_raw
    CleanedJets["p4","pt"] = CleanedJets.pt_raw
    CleanedJets["p4","mass"] = CleanedJets.mass_raw
    if (isMC):
        CleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(CleanedJets.matched_gen.pt, 0), np.float32)
    debug(self.debugMode, "Raw Jets (Before Cleaning):       ", CleanedJets[0].pt)

    ## (2) Removing leptons from jets

    if leptons_inJet != None :
        cleaned = (CleanedJets.p4).subtract(leptons_inJet)
        CleanedJets["p4","pt"] = cleaned.pt
        CleanedJets["p4","eta"] = cleaned.eta
        CleanedJets["p4","phi"] = cleaned.phi
        CleanedJets["pt"] = cleaned.pt
        CleanedJets["eta"] = cleaned.eta
        CleanedJets["phi"] = cleaned.phi
        CleanedJets["pt_raw"] = cleaned.pt
    
    jec_stack = JECStack(jec_inputs)
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetEta'] = 'eta'
    name_map['JetPhi'] = 'phi'
    name_map['JetMass'] = 'mass'
    name_map['Rho'] = 'rho'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['METpt'] = 'pt'
    name_map['METphi'] = 'phi'
    name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
    name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'
    if corr_type=='met': return CorrectedMETFactory(name_map)
    CleanedJets = CorrectedJetsFactory(name_map, jec_stack).build(CleanedJets, lazy_cache=df.caches[0])    
    return CleanedJets




## --------------------------------- MET Filters ------------------------------#
## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL

MET_filters = {'2016APV':["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2016'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2017'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"],
               '2018'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"]}

corrlib_namemap = {
    "2016APV":"2016preVFP_UL",
    "2016":"2016postVFP_UL",
    "2017":"2017_UL",
    "2018":"2018_UL"
}


def GetPUSF(IOV, nTrueInt, var='nominal'):
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM

    fname = "jsonpog-integration/POG/LUM/" + corrlib_namemap[IOV] + "/puWeights.json.gz"
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    return evaluator[hname[IOV]].evaluate(np.array(nTrueInt), var)

def GetL1PreFiringWeight(IOV, df, var="Nom"):
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = ak.ones_like(df.event)
    if ("L1PreFiringWeight" in ak.fields(df)):
        L1PrefiringWeights = df["L1PreFiringWeight"][var]
    return L1PrefiringWeights

def HEMCleaning(IOV, JetCollection):
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (IOV == "2018"):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)

    return isHEM

def GetEleSF(IOV, wp, eta, pt, var = ""):
    ## Reference:
    ##   - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018
    ##   - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaSFJSON
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
    fname = "jsonpog-integration/POG/EGM/" + corrlib_namemap[IOV] + "/electron.json.gz"
    year = {
        "2016APV" : "2016preVFP",
        "2016"    : "2016postVFP",
        "2017"    : "2017",
        "2018"    : "2018",
    }
    num = ak.num(pt)
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    
    ## if the eta and pt satisfy the requirements derive the eff SFs, otherwise set it to 1.
    mask = pt > 20
    pt = ak.where(mask, pt, 22)
    
    sf = evaluator["UL-Electron-ID-SF"].evaluate(year[IOV], "sf"+var, wp,
                                                 np.array(ak.flatten(eta)),
                                                 np.array(ak.flatten(pt)))
    sf = ak.where(np.array(ak.flatten(~mask)), 1, sf)
    return ak.unflatten(sf, ak.num(pt))

def GetMuonSF(IOV, corrset, abseta, pt, var="sf"):
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonReferenceEffsRun2
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO
    ## var = "sf", "systup", "systdown"
    fname = "jsonpog-integration/POG/MUO/" + corrlib_namemap[IOV] + "/muon_Z.json.gz"
    if "reco" in corrset:
        hname = "NUM_TrackerMuons_DEN_genTracks"
    else:
        hname = "NUM_TightID_DEN_genTracks"
    year = {
        "2016APV" : "2016preVFP_UL",
        "2016"    : "2016postVFP_UL",
        "2017"    : "2017_UL",
        "2018"    : "2018_UL",
    } 
    num = ak.num(pt)
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    ## if the abseta and pt satisfy the requirements derive the eff SFs, otherwise set it to 1.
    
    mask = (pt > 15) & (abseta < 2.4)
    pt = ak.where(mask, pt, 22)
    abseta = ak.where(mask, abseta, 2.3)
    
    sf = evaluator[hname].evaluate(year[IOV], np.array(ak.flatten(abseta)),
                                   np.array(ak.flatten(pt)), var)
    sf = ak.where(np.array(ak.flatten(~mask)), 1, sf)
    return ak.unflatten(sf, ak.num(pt))

def GetMuonTrigEff(IOV, abseta, pt, var="nominal"):
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonReferenceEffsRun2
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO
    # var = "nominal", "up", "down"
    num = ak.num(pt)
    
    year = "2016_UL_HIPM" if IOV == '2016APV' else IOV+"_UL"
    fname = "data/muonTrigSF/Efficiencies_muon_generalTracks_Z_Run" + year + "_SingleMuonTriggers_schemaV2.json.gz"
    fname = "jsonpog-integration/POG/MUO/" + corrlib_namemap[IOV] + "/muon_Z.json.gz"
    hname = {
        "2016APV": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        "2016"   : "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        "2017"   : "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        "2018"   : "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    
    ## if the abseta and pt satisfy the requirements derive the eff SFs, otherwise set it to 1.
    
    mask = np.array(ak.flatten((pt < 200) & (pt > 52) & (abseta < 2.4)))
    pt = np.array(ak.where(mask, np.array(ak.flatten(pt)), 53))
    abseta = np.array(ak.where(mask, np.array(ak.flatten(abseta)), 2.3))
    
    sf = evaluator[hname[IOV]].evaluate(abseta, pt, "nominal")
    syst = evaluator[hname[IOV]].evaluate(abseta, pt, "syst")
    
    if var == "up":
        sf += syst
    elif var == "down":
        sf -= syst
        
    sf = ak.where(~mask, 1, sf)
    
    return ak.unflatten(sf, num)

def GetPDFweights(df, var="nominal"):
    ## determines the pdf up and down variations
    pdf = ak.ones_like(df.event)
    if ("LHEPdfWeight" in ak.fields(df)):
        pdfUnc = ak.std(df.LHEPdfWeight,axis=1)/ak.mean(df.LHEPdfWeight,axis=1)
    if var == "up":
        pdf += pdfUnc
    elif var == "down":
        pdf -= pdfUnc
    return pdf

def GetQ2weights(df, var="nominal"):
    ## determines the envelope of the muR/muF up and down variations
    ## Case 1:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 1)
    ##               [5] -> (1, 2)
    ##               [6] -> (2, 0.5)
    ##               [7] -> (2, 1)
    ##               [8] -> (2, 2)
                  
    ## Case 2:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 2)
    ##               [5] -> (2, 0.5)
    ##               [6] -> (2, 1)
    ##               [7] -> (2, 2)

    q2 = ak.ones_like(df.event)
    q2Up = ak.ones_like(df.event)
    q2Down = ak.ones_like(df.event)
    if ("LHEScaleWeight" in ak.fields(df)):
        if ak.all(ak.num(df.LHEScaleWeight)==9):
            nom = df.LHEScaleWeight[:,4]
            scales = df.LHEScaleWeight[:,[0,1,3,5,7,8]]
            q2Up = ak.max(scales,axis=1)/nom
            q2Down = ak.min(scales,axis=1)/nom 
        elif ak.all(ak.num(df.LHEScaleWeight)==9):
            scales = df.LHEScaleWeight[:,[0,1,3,4,6,7]]
            q2Up = ak.max(scales,axis=1)
            q2Down = ak.min(scales,axis=1)
            
    if var == "up":
        return q2Up
    elif var == "down":
        return q2Down
    else:
        return q2

    
def getLumiMaskRun2():

    golden_json_path_2016 = "data/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
    golden_json_path_2017 = "data/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
    golden_json_path_2018 = "data/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

    masks = {"2016APV":LumiMask(golden_json_path_2016),
             "2016":LumiMask(golden_json_path_2016),
             "2017":LumiMask(golden_json_path_2017),
             "2018":LumiMask(golden_json_path_2018)
            }

    return masks
