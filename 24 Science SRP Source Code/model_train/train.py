import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# data inport
potassium_iodide = np.array([2, 3, 4, 5, 6])
averages_sigmoid = np.array([12.0, 14.0, 23.0, 30.0, 32.0])
standard_deviations = np.array([1.53, 1.0, 4.58, 4.36, 4.51])  # Standard deviations for error range

# sig function definition
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

# init parameter guess [L, x0, k, b]
initial_guess = [30, 4, 1, 10]

# fitting sigmoid curve
popt, pcov = curve_fit(sigmoid, potassium_iodide, averages_sigmoid, p0=initial_guess)

# getting paras
L, x0, k, b = popt
print(f"Fitted parameters:\nL = {L:.2f}\nx0 = {x0:.2f}\nk = {k:.2f}\nb = {b:.2f}")

# gen x y vals for the fitted curve
x_values = np.linspace(min(potassium_iodide), max(potassium_iodide), 100)
y_values_sigmoid = sigmoid(x_values, *popt)

# plotting
plt.figure(figsize=(8, 6))
plt.errorbar(potassium_iodide, averages_sigmoid, yerr=standard_deviations, fmt='o', color='red', capsize=5, label='Data Points & Error Range')
plt.plot(x_values, y_values_sigmoid, label='Fitted Sigmoid Curve', color='blue')
plt.title('Correlation Between Potassium Iodide Concentration and Foam Production \n in the Catalytic Decomposition of Hydrogen Peroxide')
plt.xlabel('Amount of Potassium Iodide (mL)')
plt.ylabel('Produced Foam Volume Increase (mL)')
plt.legend()
plt.grid(True)
plt.show()

