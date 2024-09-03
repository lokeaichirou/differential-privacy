from ..statistical_tools.quantiles import quantile


def median(array,
           privacy_notion,
           bounds       = None,
           axis         = None,
           keepdims     = False,
           random_state = None,
           accountant   = None,
           **unused_args):

    return quantile(array,
                    quant           = 0.5,
                    privacy_notion  = privacy_notion,
                    bounds          = bounds,
                    axis            = axis,
                    keepdims        = keepdims,
                    random_state    = random_state,
                    single_querying = True,
                    accountant      = accountant,
                    unused_args     = unused_args)
