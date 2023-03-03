from smp_utils import *
from response_maker_nanov9_lib import *
from response_maker_nanov9 import *
import sys

def main(argv):
    
    response_maker_nanov9(client=None, do_gen=True, testing=False, prependstr="/mnt/data/cms/")
#    response_maker_nanov9(client=None, do_gen=False, testing=False, prependstr="/mnt/data/cms/")
    
if __name__ == "__main__":
    main(sys.argv)
