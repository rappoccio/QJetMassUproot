from dask.distributed import Client
from smp_utils import *
from response_maker_nanov9_lib import *
from response_maker_nanov9 import *

def main():

    client = Client("tls://rappoccio-40gmail-2ecom.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
    
    response_maker_nanov9(client=client, do_gen=True, testing=False)
    response_maker_nanov9(client=client, do_gen=False, testing=False)
    
if __name__ == "__main__":
    main()