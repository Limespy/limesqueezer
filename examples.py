'''
Examples
========================================================================

A series of examples on how to use this package
'''
import pathlib
PATH_REPO = pathlib.Path(__file__).parent
PATH_FIGURES = PATH_REPO / 'figures'

#%%═════════════════════════════════════════════════════════════════════
# QUICK START BLOCK

import numpy as np

import limesqueezer as  ls
x_data = np.linspace(0, 1, int(1e4))
y_data = np.sin(24 * x_data ** 2)

tolerance = 0.05
x_compressed, y_compressed = ls.compress(x_data, y_data, tolerances = tolerance)

x0, y0 = x_data[0], y_data[0]
generator = zip(x_data[1:], y_data[1:])

with ls.Stream(x0, y0, tolerances = tolerance) as record:
    for x_value, y_value in generator:
        record(x_value, y_value)

x_compressed, y_compressed = record.x, record.y

# Decompression
function = ls.decompress(x_compressed, y_compressed)
y_decompressed = function(x_data).reshape(y_data.shape)

residuals = y_decompressed - y_data
maximum_error = np.amax(np.abs(residuals))
print(f'Maximum error should be ~= {tolerance}: {maximum_error:.5f}')

from matplotlib import pyplot as plt

fig, axs = plt.subplots(2,1, sharex=True)
# Data and compressed
axs[0].plot(x_data, y_data, label='Original')
axs[0].plot(x_compressed, y_compressed, '-o', label ='Compressed')
axs[0].legend()

# Residuals to tolerance
residuals = y_decompressed - y_data
axs[1].plot(x_data, y_decompressed - y_data, label = 'Residuals')
axs[1].axhline(tolerance, label = 'Total tolerance', color = 'red')
axs[1].axhline(-tolerance, color = 'red')
axs[1].legend()

fig.tight_layout()
# Instead of showing the figure it is saved as png
plt.savefig(PATH_FIGURES / 'quick_start.png', bbox_inches = 'tight')

#%%═════════════════════════════════════════════════════════════════════
# DIFFERENT ERROR TOLERANCE
# The first example was using absolute tolerance.
def plot_data_compressed_decompressed_1d(x_compressed,
                                         y_compressed,
                                         y_decompressed,
                                         total_tolerance,
                                         residuals,
                                         fname,
                                         ylim = (None, None)):

    fig, axs = plt.subplots(2,1, sharex = True)
    # Data and compressed
    axs[0].plot(x_data, y_data, color = 'blue', label = 'Original')
    axs[0].plot(x_compressed, y_compressed, 'o', color = 'orange', label = 'Compressed')
    axs[0].plot(x_data, y_decompressed, color = 'orange')
    axs[0].grid(True)
    axs[0].legend(loc = 'lower left')

    axs[1].plot(x_data, residuals, label = 'Residuals')
    axs[1].plot(x_data, total_tolerance,
                label = 'Total tolerance', color = 'red')
    axs[1].plot(x_data, -total_tolerance, color = 'red')
    axs[1].grid(True)
    axs[1].legend(loc = 'lower left')

    if ylim[0]:
        axs[0].set_ylim(ylim[0])
    if ylim[1]:
        axs[1].set_ylim(ylim[1])

    fig.tight_layout()
    # Instead of showing the figure it is saved as png
    if fname:
        plt.savefig(PATH_FIGURES / (fname + '.png'), bbox_inches = 'tight')
#───────────────────────────────────────────────────────────────────────
def plot_tolerances(tolerances, fname):
    x_compressed, y_compressed = ls.compress(x_data, y_data,
                                             tolerances = tolerances)
    function = ls.decompress(x_compressed, y_compressed)
    y_decompressed = function(x_data).reshape(y_data.shape)
    residuals = y_decompressed - y_data
    total_tolerance = ls.tolerancefunctions[0](y_data, tolerances)
    plot_data_compressed_decompressed_1d(x_compressed,
                                         y_compressed,
                                         y_decompressed,
                                         total_tolerance,
                                         residuals,
                                         fname,
                                         ylim = (None, (-0.05, 0.05)))
##%%════════════════════════════════════════════════════════════════════
## Absolute only
relative = 0
absolute = 0.02
falloff = 0
plot_tolerances((relative, absolute, falloff),
                'absolute_only')
##%%════════════════════════════════════════════════════════════════════
## Relative only
relative = 0.03
absolute = 0
falloff = 0
plot_tolerances((relative, absolute, falloff),
                'relative_only')
##%%════════════════════════════════════════════════════════════════════
## Relative and absolute with zero falloff
relative = 0.03
absolute = 0.02
falloff = 0
plot_tolerances((relative, absolute, falloff),
                'relative_and_absolute_no_falloff')
##%%════════════════════════════════════════════════════════════════════
## Relative and absolute with smooth falloff
relative = 0.03
absolute = 0.02
falloff = relative / absolute
plot_tolerances((relative, absolute, falloff),
                'relative_and_absolute_smooth_falloff')
##%%════════════════════════════════════════════════════════════════════
## Relative and absolute with too much falloff
relative = 0.03
absolute = 0.02
falloff = relative / absolute * 4
plot_tolerances((relative, absolute, falloff),
                'relative_and_absolute_over_falloff')
#%%═════════════════════════════════════════════════════════════════════
# ERROR FUNCTIONS
#───────────────────────────────────────────────────────────────────────
def plot_errorfunction1(errorfunction):
    tolerances = (0, 0.02, 0)
    x_compressed, y_compressed = ls.compress(x_data, y_data,
                                             tolerances = tolerances,
                                             errorfunction = errorfunction)
    function = ls.decompress(x_compressed, y_compressed)
    y_decompressed = function(x_data).reshape(y_data.shape)
    residuals = y_decompressed - y_data
    total_tolerance = ls.tolerancefunctions[0](y_data, tolerances)
    plot_data_compressed_decompressed_1d(x_compressed,
                                         y_compressed,
                                         y_decompressed,
                                         total_tolerance,
                                         residuals,
                                         errorfunction,
                                         ylim = (None, (-0.05, 0.05)))
##%%════════════════════════════════════════════════════════════════════
## MAXABS
# Same as absolute only error tolerance
plot_errorfunction1('MaxMAbs')
plot_errorfunction1('MaxMAbs_AbsEnd')

##%%════════════════════════════════════════════════════════════════════
## MAXMS
def plot_errorfunction2(errorfunction):
    tolerances = (0, 0.02, 0)
    x_compressed, y_compressed = ls.compress(x_data, y_data,
                                             tolerances = tolerances,
                                             errorfunction = errorfunction)
    function = ls.decompress(x_compressed, y_compressed)
    y_decompressed = function(x_data).reshape(y_data.shape)
    residuals = y_decompressed - y_data
    residuals *= np.abs(residuals)
    total_tolerance = ls.tolerancefunctions[0](y_data, tolerances)
    plot_data_compressed_decompressed_1d(x_compressed,
                                         y_compressed,
                                         y_decompressed,
                                         total_tolerance,
                                         residuals,
                                         errorfunction,
                                         ylim = (None, (-0.05, 0.05)))
# Maximum of mean squares
plot_errorfunction2('MaxMS')
##%%════════════════════════════════════════════════════════════════════
## MAXMS_SEND
# Maximum of mean squares or maximum of end sqruared
plot_errorfunction2('MaxMS_SEnd')
# #%%═════════════════════════════════════════════════════════════════════
# # STREAM

# example_x0, example_y0 = X_DATA[0], Y_DATA[0]
# generator = zip(X_DATA[1:], Y_DATA[1:])

# # The context manager for Stream data is 'Stream'.

# with ls.Stream(example_x0, example_y0, tolerances = tolerances, errorfunction = 'MaxAbs') as record:
#     # A side mote: In Enlish language the word 'record' can be either
#     # verb or noun and since it performs this double role of both taking
#     # in data and being storage of the data, it is a fitting name for the object

#     # For sake of demonstarion
#     print(f'Record state within the context manager is {record.state}')
#     for example_x_value, example_y_value in generator:
#         record(example_x_value, example_y_value)

#     # Using record.x or record.y in the with statement block results in
#     # attribute error, as those attributes are generated only when
#     # the record is closed.
#     #
#     # If you want to access the already compressed data
#     x_compressed, y_compressed = record.xc, record.yc
#     # and to access the buffered data waiting more values or closing of
#     # the record to be compressed.
#     x_buffered, y_buffered = record.xb, record.yb

# x_compressed, y_compressed = record.x, record.y
# print(f'Record state after the context manager is {record.state}' )

# # Side Note:
# # There is a third state called 'closing'.
# # In case the record closing process crashes somehow, the state is left to this.

# #%%═════════════════════════════════════════════════════════════════════
# # DECOMPRESSION

# function = ls.decompress(x_compressed, y_compressed)
# y_decompressed = function(X_DATA).flatten()
