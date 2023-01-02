import tensorflow as tf

class GibbsParameter:
    def __init__(self, value, conditional_posterior, posterior_params=None):
        self.value = value
        self.conditional_posterior = conditional_posterior
        self.posterior_params = posterior_params

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return str(self.value)

    def sample(self, sample_params):
        param_values = {}
        for k, v in sample_params.items():
            if isinstance(v, GibbsParameter):
                param_values[k] = v.value
            else:
                param_values[k] = v
        post_params = param_values
        self.value = self.conditional_posterior(post_params)
        return self.value


class GibbsSampler:
    def __init__(self, params):
        self.params = params

    def single_sample(self, param_name):
        value = self.params[param_name].sample(self.params)
        self.params[param_name].value = value
        return value

    @tf.function
    def sampling_routine(
        self,
        num_samples,
        sample_period=1,
        sample_burnin=0,
        sample_thining=1,
        printRetraceFlag=True,
    ):
        if printRetraceFlag:
            print("retracing")

        params = self.params
        history = []
        step_num = sample_burnin + num_samples * sample_thining
        for n in range(step_num):
            row = {}
            for key in list(params.keys()):
                if isinstance(params[key], GibbsParameter):
                    row[key] = self.single_sample(key)
            if (n >= sample_burnin) & (n % sample_period == 0):
                history.append(row)
        return history
