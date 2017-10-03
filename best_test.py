import pymc3 as pm

class BayesianDifferenceTest:
    
    """Perform a Bayesian test for difference of means and standard deviations between two samples.
       Inspired by the classic BEST paper by Kruschke.
    """

    def __init__(self, param_a_name, param_b_name, mu_mean, mu_sd, sd_lower, sd_upper, nu_mean):
        
        """Initialize the test.
        
           Parameters
           ----------
            param_a_name, param_b_name : str
                Names for the tested parameters.
            mu_mean, mu_sd : float
                Mean and standard deviation of the prior on mean.
            sd_upper, sd_lower : float
                Upper and lower bounds of the prior on standard deviation.
            nu_mean : float
                The mean of the prior on normality (aka 'degrees of freedom').
        """
        
        self.param_a_name = param_a_name
        self.param_b_name = param_b_name
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.sd_lower = sd_lower
        self.sd_upper = sd_upper
        self.nu_mean = nu_mean
        
        self._varnames = {
            'mean_param_a': '{}_mean'.format(self.param_a_name),
            'mean_param_b': '{}_mean'.format(self.param_b_name),
            'sd_param_a'  : '{}_sd'.format(self.param_a_name),
            'sd_param_b'  : '{}_sd'.format(self.param_b_name),
            'nu'          : 'nu',
            'diff_means'  : 'difference_of_means',
            'diff_sds'    : 'difference_of_sds'
        }
    
    def _build_model(self, observed_a, observed_b):
        self.model = pm.Model()
        with self.model as model:
            # normal priors for means
            mean_param_a = pm.Normal(self._varnames['mean_param_a'], self.mu_mean, self.mu_sd)
            mean_param_b = pm.Normal(self._varnames['mean_param_b'], self.mu_mean, self.mu_sd)

            # uniform priors standard deviations
            sd_param_a = pm.Uniform(self._varnames['sd_param_a'], self.sd_lower, self.sd_upper)
            sd_param_b = pm.Uniform(self._varnames['sd_param_b'], self.sd_lower, self.sd_upper)

            # shifted exponential prior for normality (aka 'degrees of freedim')
            nu = pm.Exponential(self._varnames['nu'], 1 / self.nu_mean) + 1

            # the data is assumed to come from Student's t distribution since it models data with outliers well
            # it is not realted to Student's t test in this case

            # pymc3 uses precision instead of sd for Student's t
            lambda_param_a = sd_param_a ** -2
            lambda_param_b = sd_param_b ** -2

            data_param_a = pm.StudentT('data_param_a', nu=nu, mu=mean_param_a, lam=lambda_param_a, observed=observed_a)
            data_param_b = pm.StudentT('data_param_b', nu=nu, mu=mean_param_b, lam=lambda_param_b, observed=observed_b)
            
            diff_means = pm.Deterministic(self._varnames['diff_means'], mean_param_a - mean_param_b)
            diff_sds = pm.Deterministic(self._varnames['diff_sds'], sd_param_a - sd_param_b)
    
    def run(self, observed_a, observed_b, nsamples=2000, njobs=1):
        
        """Run the inference on the model.
        
           Parameters
           ----------
           observed_a, observed_b : array-like
               The observed data for the test.
           nsamples : int, optional
               The number of samples for MCMC (default 2000).
           njobs : int, optional
               the number of concurrent processes to use for sampling (default 1).
        """
        
        self._build_model(observed_a, observed_b)
        with self.model as model:
            self.trace = pm.sample(nsamples, njobs=njobs)
            
        
    def plot_posterior(self, varnames=None, ref_val=None):
        
        """Generate informative plots form the trace.
        
           Parameters
           ----------
           varnames : iterable of str or None, optional
               The model variables to generate plots for (default None).
               If None, defaults to all variables.
           ref_val: int or float or None, optional
               The value to use as reference on the plots (default None).
               Generally only relevant for posteriors on differences of means 
               and standard deviations. For example, if ref_val = 0, a bar will
               be placed on the posterior plot at a point corresponding to
               zero difference in parameters. If this bar lies within the 95% HPD,
               then it is likely that there is no significant difference between 
               the parameters.
        """
        
        varnames = varnames or self.model_variables
        pm.plot_posterior(self.trace, varnames=varnames, ref_val=ref_val, color='#8BCAF1')
        
    def forestplot(self, varnames=None):
        
        """Generate a forestplot with 95% credible intervals and R hat statistic.
        
           Parameters
           ----------
           varnames : iterable of str or None, optional
               The model variables to generate plots for (default None).
               If None, defaults to all variables.
        """
        
        varnames = varnames or self.model_variables
        pm.forestplot(self.trace, varnames=varnames, color='#8BCAF1')
        
    def traceplot(self):
        
        """Generate a traceplot for MCMC diagnostics."""
        
        pm.traceplot(self.trace)
        
    def summary(self, varnames=None):
        
        """Generate summary statistics for model as Pandas dataframe.
        
           Parameters
           ----------
           varnames : iterable of str or None, optional
               The model variables to generate summaries for (default None).
               If None, defaults to all variables.
               
          Returns
          -------
          summary : pandas.DataFrame
              The dataframe with summary statistics.
        """
        
        varnames = varnames or self.model_variables
        return pm.df_summary(self.trace, varnames=varnames)
        
    @property
    def model_variables(self):
        
        """Get model variables.
           
           Returns
           -------
           varnames : list of str
               The names of model variables.
        """
        
        return list(self._varnames.values())
