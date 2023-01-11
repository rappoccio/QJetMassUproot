# Can grab a file on cmslpc from 
# /store/group/lpctlbsm/NanoAODJMAR_2019_V1/Production/CRAB/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/DYJetsToLLM-50TuneCUETP8M113TeV-madgraphMLM-pythia8RunIISummer16MiniAODv3-PUMoriond17_ext2-v2/190513_171710/0000/*.root
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
import glob
import pickle

from response_maker_nanov9_lib import *


def main(testing=False, do_gen=True): 

    if not testing: 
        nworkers = 8
        chunksize = 1000000
        maxchunks = None
    else:
        nworkers = 1
        chunksize = 10
        maxchunks = 1
    
    
    fileset = {}
    if do_gen:
        eras = [
            'UL16NanoAOD', 
            'UL16NanoAODAPV', 
            'UL17NanoAOD', 
            'UL18NanoAOD'
               ]
        htbins = ['100to200', '200to400', '400to600', '600to800', '800to1200', '1200to2500', '2500toInf']
        filestr = '/mnt/data/cms/store/mc/RunIISummer20%sv9/DYJetsToLL_M-50_HT-%s_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/*.root'
    
        for era in eras: 
            for htbin in htbins : 
                infiles = glob.glob(filestr % (era, htbin) )
                if testing: 
                    infiles = infiles[0:2]
                binname = era+htbin
                if binname not in fileset:
                    fileset[binname] = []
                fileset[binname] = fileset[binname] + [*infiles]
                
    else :
        eras = [ 
            'Run2016B',  'Run2016C',  'Run2016D',  'Run2016E',  'Run2016F',  'Run2016G',  'Run2016H',  
            'Run2017B',  'Run2017C',  'Run2017D',  'Run2017E',  'Run2017F',  
            'Run2018A',  'Run2018B',  'Run2018C',  'Run2018D'
        ]
        filestr1 = '/mnt/data/cms/store/data/%s/SingleMuon/NANOAOD/*/*/*.root'
        filestr2 = '/mnt/data/cms/store/data/%s/SingleElectron/NANOAOD/*/*/*.root'
        filestr3 = '/mnt/data/cms/store/data/%s/EGamma/NANOAOD/*/*/*.root'
        
        for era in eras:
            muinfiles = glob.glob( filestr1 % (era) )
            e1infiles = glob.glob( filestr2 % (era) )
            e2infiles = glob.glob( filestr3 % (era) )
            if testing:
                filestr1 = filestr1[0:2]
                filestr2 = filestr2[0:2]
                filestr3 = filestr3[0:2]
            if era not in fileset:
                fileset[era] = []
            fileset[era] = fileset[era] + [*muinfiles] + [*el1infiles] + [*el2infiles]
        

    print("Processing files ")
    for era,files in fileset.items():
        print(era)
        for file in files:
            print(file)



    tstart = time.time() 

    run = processor.Runner(
        executor = processor.FuturesExecutor(compression=None, workers=nworkers),
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=maxchunks
    )

    output = run(
        fileset,
        "Events",
        processor_instance=QJetMassProcessor(do_gen=do_gen),
    )
    
    if do_gen:
        fname_out = 'qjetmass_zjets_gen.pkl'
    else:
        fname_out = 'qjetmass_zjets_reco.pkl'
    with open(fname_out, "wb") as f:
        pickle.dump( output, f )


    
if __name__ == "__main__":
    main()