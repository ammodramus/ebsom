from collections import Counter

class GroupCollector(object):
    '''
    Collects and tabulates observations that can be grouped together
    
    Examples include, for non-candidate loci, the observations for each (consensus, read, observed-based) tuple,
    or, for each candidate locus
    '''
    def __init__(self, rowlen):
        self.rowcounts = Counter()
        self.rowlen = rowlen
    
    def add(self, row):
        if row.shape[0] != rowlen:
            raise ValueError('incorrect row length')
        self.rowcounts[row.data] += 1
    
    def collect(self):
        all_rows = []
        all_counts = []
        for buf, count in self.rowcounts.iteritems():
            all_rows.append(np.frombuffer(buf))
            all_counts.append(count)
        return np.array(all_rows), np.array(all_counts)


class CandidateCollector(object):
    '''
    takes a row and a candidate locus, adds 
    '''
    def __init__(self, rowlen):
        self.rowlen = rowlen
        self.groups = {}  # keys are ref, refpos, is_reverse, (consbase, readnum, obsbase)
    
    def add(self, row, ref, reflocus, is_reverse, consbase, readnum, obsbase):
        d = self.groups
        # three layers of dicts before the final dict
        for key in (ref, reflocus, is_reverse):
            try:
                d = d[key]
            except KeyError:
                d[key] = {}
                d = d[key]
        final_key = (consbase, readnum, obsbase)
        try:
            d[final_key].add(row)
        except KeyError:
            thisgc = GroupCollector(self.rowlen)
            d[final_key] = thisgc
            thisgc.add(row)
    
    def collect(self):
        '''
        returns a dict d, where

        d[ref][loc][is_reverse][(consbase,readnum,obsbase)]

        is the tuple (X, c), where X is a covariate matrix, c are this matrices
        multiplicities, ref is the reference name, loc is the position,
        is_reverse is boolean, and (consbase,readnum,obsbase) is the final key.
        '''
        ret = {}
        for ref, refdict in self.groups.iteritems():
            ret[ref] = {}
            for loc, locdict in refdict.iteritems():
                ret[ref][loc] = {}
                for rev, revdict in ret[ref][loc].iteritems():
                    ret[ref][loc][rev] = {}
                    # "final key" == (cons,rnum,obs)
                    for fkey, group in revdict.iteritems():
                        ret[ref][loc][ref][fkey] = group.collect()
        return ret


class NonCandidateCollector(object):
    '''
    takes a row and a candidate locus, adds 
    '''
    def __init__(self, rowlen):
        self.rowlen = rowlen
        self.groups = {}  # keys are ref, refpos, is_reverse, (consbase, readnum, obsbase)
    
    def add(self, row, consbase, readnum, obsbase):
        key = (consbase, readnum, obsbase)
        try:
            gc = self.groups[key]
        except KeyError:
            gc = GroupCollector(self.rowlen)
            self.groups[key] = gc
        gc.add(row)
    
    def collect(self):
        '''
        returns a dict d, where

        d[(consbase,readnum,obsbase)]

        is the tuple (X, c), where X is a covariate matrix, c are the row
        multiplicities.
        '''
        ret = {}
        for ref, refdict in self.groups.iteritems():
            ret[ref] = {}
            for loc, locdict in refdict.iteritems():
                ret[ref][loc] = {}
                for rev, revdict in ret[ref][loc].iteritems():
                    ret[ref][loc][rev] = {}
                    # "final key" == (cons,rnum,obs)
                    for fkey, group in revdict.iteritems():
                        ret[ref][loc][ref][fkey] = group.collect()
        return ret
