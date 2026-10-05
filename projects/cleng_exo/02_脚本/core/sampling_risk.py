"""Conservative stratified loss bounds for the frozen, random three-pool audit."""
from __future__ import annotations
from statistics import NormalDist
from core.topic_loop import wilson


def risk_bounds(metrics, confidence=.90):
    # Three proportions, six tails; Bonferroni gives at least the requested
    # simultaneous coverage under the Wilson approximation (not an exact bound).
    z = NormalDist().inv_cdf(1-(1-confidence)/6)
    bounds = {}; estimates = {}
    for pool in ('keep','drop','review'):
        m = metrics[pool]; n = m['labeled']; size = m['population']
        success = m['T'] + (m['U'] if pool=='drop' else 0)
        if size == 0:
            interval = [0.,0.]; estimate = 0.
        elif not n:
            interval = [0.,1.]; estimate = None
        elif n == size:
            estimate = success/n; interval = [estimate,estimate]
        else:
            estimate = success/n; interval = wilson(success,n,z=z)
        bounds[pool] = [size*x for x in interval]
        estimates[pool] = None if estimate is None else size*estimate
    d,k,r = (bounds[p] for p in ('drop','keep','review'))
    low_den = d[0]+k[1]+r[1]; high_den = d[1]+k[0]+r[0]
    lo = d[0]/low_den if low_den else 0.
    hi = d[1]/high_den if high_den else (0. if metrics['drop']['population']==0 else 1.)
    total = sum(estimates.values()) if all(v is not None for v in estimates.values()) else None
    return {'metric':'drop_T_plus_U_over_drop_T_plus_U_and_retained_T',
            'estimate':estimates['drop']/total if total else None, 'interval':[lo,hi],
            'estimated_counts':estimates,'count_bounds':bounds,'confidence':confidence,
            'method':'stratified_counts_with_bonferroni_wilson_or_census',
            'z':z,'note':'Drop U counts as potential lost T. Other U does not enlarge the denominator. Historical/nonrandom samples do not estimate a new batch.'}
