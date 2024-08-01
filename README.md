# Machine and Deep Learning on the Structural Connectectome

## Running DL Experiments
```console
python3 dl_classification.py
```

> Command line args (defaults **bolded**):  

-- dataset : {bright, DHCP, **combined**}  
&emsp; bright = adult cohort, DHCP = pediatric cohort  
-- device : {**cpu**, gpu}  
-- n_nodes : {76, **379**}  
&emsp; 76 = working memory network, 379 = whole brain network  
-- model : {**GCNConv**, GCN, NN}  
&emsp; GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron  
-- seed : int, default=**123**  
-- epochs : int, default=**100**  
-- lr : float, default=**1e-3**

> python3 dl_regression.py

Command line args (defaults **bolded**):
- --dataset (**bright** or DHCP)
   - bright = adult cohort, DHCP = pediatric cohort 
- --device (**cpu** or gpu)
- --n_nodes (76 or **379**)
   - 76 = working memory network, 379 = whole brain network
- --score_type (**CPTOmZs**, CodingZs, DigitBwdSpanZs, lswm, dccs, pcps)
   - Bright scores: CPTOmZs (sustained attention), CodingZs (processing speed), DigitBwdSpanZs (executive function)
   - DHCP scores: lswm (working memory), dccs (executive function), pcps (processing speed)
- --model (**GCNConv**, GCN, or NN)
   - GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron
- --seed (default **123**)
- --epochs (default **100**)
- --lr (default **1e-2**)


## Running ML Experiments
### Sex Classification
> python3 ml_classification.py

Command line args (defaults **bolded**):
- --dataset (bright, DHCP, or **combined**)
   - bright = adult cohort, DHCP = pediatric cohort
- --n_nodes (76 or **379**)
   - 76 = working memory network, 379 = whole brain network
- --n_components (default **100**)
   - number of output components for PCA dimensionality reduction
- --seed (default **123**)

> python3 ml_regression.py

Command line args (defaults **bolded**):
- --dataset (bright, DHCP, or **combined**)
   - bright = adult cohort, DHCP = pediatric cohort
- --n_nodes (76 or **379**)
   - 76 = working memory network, 379 = whole brain network
- --score_type (**CPTOmZs**, CodingZs, DigitBwdSpanZs, lswm, dccs, pcps)
   - Bright scores: CPTOmZs (sustained attention), CodingZs (processing speed), DigitBwdSpanZs (executive function)
   - DHCP scores: lswm (working memory), dccs (executive function), pcps (processing speed)
- --n_components (default **100**)
   - number of output components for PCA dimensionality reduction
- --seed (default **123**)
