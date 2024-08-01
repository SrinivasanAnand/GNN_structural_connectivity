# Machine and Deep Learning on the Structural Connectectome

## Running Deep Learning Experiments
### Sex Classification
```console
python3 dl_classification.py
```

> Command line args:  

-- **dataset : {bright, DHCP, combined}, default=combined**  
&emsp;&emsp; bright = adult cohort, DHCP = pediatric cohort  
-- **device : {cpu, gpu}, default=cpu**  
-- **n_nodes : {76, 379}, default=379**  
&emsp;&emsp; 76 = working memory network, 379 = whole brain network  
-- **model : {**GCNConv**, GCN, NN}, default=GCNConv**  
&emsp;&emsp; GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron  
-- **seed : *int*, default=123**   
-- **epochs : *int*, default=100**  
-- **lr : *float*, default=1e-3**

```console
python3 dl_regression.py
```
### Cognitive Score Regression

> Command line args:

-- **dataset : {bright, DHCP}, default=bright**  
&emsp;&emsp; bright = adult cohort, DHCP = pediatric cohort  
-- **device : {cpu, gpu}, default=cpu**  
-- **n_nodes : {76, 379}, default=379**  
&emsp;&emsp; 76 = working memory network, 379 = whole brain network  
--**score_type {CPTOmZs, CodingZs, DigitBwdSpanZs, lswm, dccs, pcps}, default=CPTOmZs**  
&emsp;&emsp; Bright scores: CPTOmZs (sustained attention), CodingZs (processing speed), DigitBwdSpanZs (executive function)  
&emsp;&emsp; DHCP scores: lswm (working memory), dccs (executive function), pcps (processing speed)  
-- **model : {GCNConv, GCN, or NN}, default=GCNConv**  
&emsp;&emsp; GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron  
-- **seed : *int*, default=123**  
-- **epochs : *int*, default=100**  
-- **lr : *float*, default=1e-2**


## Running Machine Learning Experiments
### Sex Classification
```console
python3 ml_classification.py
```

> Command line args:

--**dataset : {bright, DHCP, combined}, default=combined**  
&emsp;&emsp; bright = adult cohort, DHCP = pediatric cohort  
--**n_nodes : {76, 379}, default=379**  
&emsp;&emsp; 76 = working memory network, 379 = whole brain network  
--**n_components : *int*, default=100**  
&emsp;&emsp; number of output components for PCA dimensionality reduction  
--**seed : *int*, default=123**

### Cognitive Score Regression
```console
python3 ml_regression.py
```

> Command line args:

--**dataset : {bright, DHCP, combined}, default=combined**  
&emsp;&emsp; bright = adult cohort, DHCP = pediatric cohort  
--**n_nodes : {76, 379}, default=379**  
&emsp;&emsp; 76 = working memory network, 379 = whole brain network  
--**score_type : {CPTOmZs, CodingZs, DigitBwdSpanZs, lswm, dccs, pcps}, default=CPTOmZs**  
&emsp;&emsp; Bright scores: CPTOmZs (sustained attention), CodingZs (processing speed), DigitBwdSpanZs (executive function)  
&emsp;&emsp; DHCP scores: lswm (working memory), dccs (executive function), pcps (processing speed)  
--**n_components : *int*, default=100**  
&emsp;&emsp; number of output components for PCA dimensionality reduction  
--**seed : *int*, default=123**
