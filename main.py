
# Name : Rahul Pandey
#Roll no. : 21213  
 #from classification2 import classification
from t3 import classification
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


clf=classification('', clf_opt='dt',
                        no_of_selected_features=20)

clf.classification()


