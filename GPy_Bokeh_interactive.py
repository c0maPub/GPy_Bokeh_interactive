# %%
# 0. Import packages

import numpy as np
import GPy

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure


# %%
# 1. Define test function
# http://infinity77.net/global_optimization/test_functions_1d.html
# Problem 7
def func(x):
    return np.sin(x) + np.sin(10 / 3 * x) + np.log(x) - 0.84 * x + 3.0


xMin = 2.7
xMax = 7.5
# %%
# 2. Sample points
nSamples = 5
X = np.array([np.random.uniform(xMin, xMax, nSamples)]).reshape(-1, 1)
Y = np.array(list(map(func, X)))
# Add noise
# Y += np.random.normal(0, 0.1, (nSamples, 1))
Y = Y.reshape(-1, 1)

# for plotting "true" function
xx = np.linspace(xMin, xMax, 1000)
yy = list(map(func, xx))
# %%
# 3. Build GPR model and predict function value
kernel = GPy.kern.RBF(input_dim=1)
GPR = GPy.models.GPRegression(X, Y, kernel, normalizer=True)
# Initial values
GPR.rbf.variance = 1.0
GPR.rbf.lengthscale = 1.0
GPR.Gaussian_noise.variance = 1.0
# GPR.rbf.variance.fix()
# GPR.rbf.lengthscale.fix()
# GPR.Gaussian_noise.fix()

# MLE fitting
GPR.optimize(messages=True, optimizer="lbfgs")
GPR.optimize_restarts(num_restarts=10)
y_qua = GPR.predict_quantiles(xx.reshape(-1, 1), quantiles=(50, 2.5, 97.5))
y_pred = y_qua[0]
lower = y_qua[1]
upper = y_qua[2]

# %%
# 4. Interactive plot with bokeh
# Prepare source
sourcePred = ColumnDataSource(data=dict(x=xx, y=y_pred))
sourceSamples = ColumnDataSource(data=dict(x=X, y=Y))
sourceTrue = ColumnDataSource(data=dict(x=xx, y=yy))
sourceUncertainty = ColumnDataSource(data=dict(x=xx, y1=lower, y2=upper))

# Set up plot
plot = figure(
    height=400,
    width=800,
    title="Gaussian process regression",
    tools="crosshair,pan,reset,save,wheel_zoom",
    x_range=[xMin, xMax],
    y_range=[
        np.min([np.array(yy).reshape(-1, 1), lower, upper]),
        np.max([np.array(yy).reshape(-1, 1), lower, upper]),
    ],
)
plot.line(
    "x",
    "y",
    source=sourceTrue,
    line_width=3,
    line_alpha=1.0,
    line_color="red",
    legend_label="True",
)
plot.line(
    "x", "y", source=sourcePred, line_width=3, line_alpha=1, legend_label="GPR",
)
plot.scatter(
    "x",
    "y",
    source=sourceSamples,
    marker="o",
    size=10,
    fill_color="black",
    legend_label="Samples",
)
plot.varea(
    x="x",
    y1="y1",
    y2="y2",
    fill_alpha=0.2,
    source=sourceUncertainty,
    legend_label="Uncertainty",
)
plot.title.text = f"log(Likelihood):{GPR.log_likelihood():.2e}"

# Set up widgets
# text = TextInput(title="title", value="Gaussian process")

plot.legend.location = "top_left"
plot.legend.click_policy = "hide"
rbf_var = Slider(
    title="rbf.variance", value=GPR.rbf.variance[0], start=1.0e-15, end=10, step=1e-6,
)
rbf_lengthscale = Slider(
    title="rbf.lengthscale",
    value=GPR.rbf.lengthscale[0],
    start=1.0e-15,
    end=10,
    step=1e-6,
)
noise_var = Slider(
    title="Gaussian_noise.variance",
    value=GPR.Gaussian_noise.variance[0],
    start=1.0e-15,
    end=1,
    step=1e-6,
)


def update_data(attrname, old, new):

    # Get the current slider values
    GPR.rbf.variance[0] = rbf_var.value
    GPR.rbf.lengthscale[0] = rbf_lengthscale.value
    GPR.Gaussian_noise.variance[0] = noise_var.value
    GPR.update_model(True)
    y_qua = GPR.predict_quantiles(xx.reshape(-1, 1), quantiles=(50, 2.5, 97.5))
    y_pred = y_qua[0]
    lower = y_qua[1]
    upper = y_qua[2]

    sourcePred.data = dict(x=xx, y=y_pred)
    sourceUncertainty.data = dict(x=xx, y1=lower, y2=upper)
    plot.title.text = f"log(Likelihood):{GPR.log_likelihood():.3e}"


for w in [rbf_var, rbf_lengthscale, noise_var]:
    w.on_change("value", update_data)


# Set up layouts and add to document
inputs = column(rbf_var, rbf_lengthscale, noise_var)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Gaussian process regression"
