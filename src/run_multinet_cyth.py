# -*- coding: utf-8 -*-
"""

"""
if __name__ == '__main__':
    
    # without build setup
    import pyximport; pyximport.install()  
    import multi_net_cython

    import matplotlib
    from time import time
    matplotlib.use('Agg')
    t00 = time()
    multi_net_cython.run_plot()
    t10 = time()
    print('Total elapsed time: {} mins'.format( round( (t10-t00)/60, 4) ) )    
    