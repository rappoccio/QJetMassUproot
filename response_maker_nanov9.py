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


def main(testing=False): 

    if not testing: 
        nworkers = 8
        chunksize = 1000000
        maxchunks = None
    else:
        nworkers = 1
        chunksize = 10000
        maxchunks = 1
    
    eras = [
        'UL16NanoAOD', 
        'UL16NanoAODAPV', 
        'UL17NanoAOD', 
        'UL18NanoAOD'
           ]
    htbins = ['70to100', '100to200', '200to400', '400to600', '600to800', '800to1200', '1200to2500', '2500toInf']
    filestr = '/mnt/data/cms/store/mc/RunIISummer20%sv9/DYJetsToLL_M-50_HT-%s_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/*.root'
    fileset = {}

    for era in eras: 
        for htbin in htbins : 
            infiles = glob.glob(filestr % (era, htbin) )
            if testing: 
                infiles = infiles[0:2]
            if era not in fileset:
                fileset[era] = []
            else: 
                fileset[era] = fileset[era] + [*infiles]


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
        processor_instance=QJetMassProcessor(),
    )
    
    with open("qjetmass_zjets.pkl", "wb") as f:
        pickle.dump( output, f )


    
if __name__ == "__main__":
    main()