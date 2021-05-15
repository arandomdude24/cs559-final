### START: Dylan Lopez
###import libraries and set up accuracy lists

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
Accuracy_List_Before_FLD = [0.5799770510613884, 0.3929432013769363, 0.7681009753298909, 0.721170395869191, 0.6581181870338497, 0.7458978772231785, 0.4974182444061962, 0.6874928284566838, 0.7729776247848535, 0.5903614457831325, 0.6916236374067699, 0.5041308089500861, 0.7812966150315548]
Accuracy_List_After_FLD = [0.5755593803786575, 0.3975903614457831, 0.7623063683304647, 0.7087205966724038, 0.6583476764199656, 0.7282271944922548, 0.4935169248422261, 0.6656339644291452, 0.7485943775100401, 0.5857142857142856, 0.6694205393000575, 0.5093516924842226, 0.7543889845094663]
Difference_between_before_andd_after_FLD = [0.004417670682730912, 0.004647160068846801, 0.005794606999426133, 0.012449799196787126, 0.00022948938611588865, 0.017670682730923648, 0.003901319563970107, 0.02185886402753856, 0.024383247274813447, 0.004647160068846912, 0.022203098106712393, 0.005220883534136522, 0.026907630522088444]
K_Value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

###Plot accuracy before FLD

plt.plot(K_Value,Accuracy_List_Before_FLD)

plt.title("Accuracy Before Applying FLD")
plt.xlabel("K Value")
plt.ylabel("Accuracy");

###Plot accuracy between before and after FLD

plt.plot(K_Value,Difference_between_before_andd_after_FLD)

plt.title("Difference of Accuracy Before and After FLD")
plt.xlabel("K Value")
plt.ylabel("Accuracy");

###Plot accuracy before FLD

plt.plot(K_Value,Accuracy_List_After_FLD)

plt.title("Accuracy After Applying FLD")
plt.xlabel("K Value")
plt.ylabel("Accuracy");

###Set up line smoothing

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
y = [0.004417670682730912, 0.004647160068846801, 0.005794606999426133, 0.012449799196787126, 0.00022948938611588865, 0.017670682730923648, 0.003901319563970107, 0.02185886402753856, 0.024383247274813447, 0.004647160068846912, 0.022203098106712393, 0.005220883534136522, 0.026907630522088444]

x_new = np.linspace(1, 25, 300)
a_BSpline = interpolate.make_interp_spline(x, y)
y_new = a_BSpline(x_new)

plt.plot(x_new, y_new)

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
spl = UnivariateSpline(x, y)
xs = np.linspace(0, 25, 1000)
plt.plot(xs, spl(xs), 'g', lw = 3)

###Plot line smoothing

plt.title("Difference of Accuracy Before and After FLD")
plt.xlabel("K Value")
plt.ylabel("Accuracy");
plt.show()

### END: Dylan Lopez