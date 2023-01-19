# MIT License

# Copyright (c) 2022 Kit Gallagher

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        print_retrace_flag=True,
    ):
        if print_retrace_flag:
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
