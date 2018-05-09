import deepdish as dd
import numpy as np

def print_(string):
    print '# ' + string

parser = argparse.ArgumentParser(
        description='remove invariable data columns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('output', help = 'output HDF5 file name')
args = parser.parse_args()

dat = dd.io.load(args.input)
try:
    cm, lo, all_majorminor, colnames = dat
    have_colnames = True
except:
    print len(dat)
    cm, lo, all_majorminor = dat
    have_colnames = False

if not have_colnames:
    print_('warning: no column names, so just using indices')

nonvariables = []
consts = []
for j in range(cm.shape[1]):
    isnonvar = np.unique(cm[:,j]).shape[0] == 1
    isconst = np.unique(cm[:,j]) == np.array([1.0])
    nonvariables.append(isnonvar)
    if isconst:
        consts.append(j)

keeper_columns = []
keeper_column_names = []
wrote_const = False
for j in range(cm.shape[1]):
    write_col = False
    if j in consts:
        if not wrote_const:
            wrote_const = True
            write_col = True
        assert j in nonvariables
    if j not in nonvariables:
        write_col = True
    if write_col:
        keeper_columns.append(j)
        if have_colnames:
            keeper_column_names.append(colnames[j])
        else:
            keeper_column_names.append('col' + str(j))

new_cm = cm[:,np.array(keeper_columns)]
data = (new_cm, lo, all_majorminor, keeper_column_names)
import warnings
with warnings.catch_warnings():
    dd.io.save(args.save_data_as, data)
