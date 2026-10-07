"""Reject non-converged or numerically unbounded ordinary logistic inference."""
import numpy as np

def require_valid_logit(result):
    if not result.mle_retvals.get('converged',False):
        raise ValueError('Logistic fit did not converge; Wald inference is invalid')
    if not np.isfinite(result.params).all() or not np.isfinite(result.bse).all():
        raise ValueError('Non-finite coefficient or standard error; inference is invalid')
    with np.errstate(over='ignore',under='ignore'):
        bounds=np.exp(np.asarray(result.conf_int()))
    if not np.isfinite(bounds).all() or (bounds==0).any():
        raise ValueError('Unbounded/underflowing odds-ratio confidence interval; inference is invalid')
