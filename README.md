# Machine and Deep Learning on the Structural Connectectome

## Running ML Experiments
python3 ml_classification.py

Command line args (defaults **bolded**):
- --dataset (bright, DHCP, or **combined**)
- --device (**cpu** or gpu)
- --n_nodes (76 or **379**)
   + 76 = working memory network, 379 = whole brain network
- --model (**GCNConv**, GCN, or NN)
   + GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron
- --seed (default 123)
- --epochs (default 100)
- --lr (default 1e-3)

## Running DL Experiments
python3 dl_classification.py

Command line args (defaults **bolded**):
- --dataset (bright, DHCP, or **combined**)
- --device (**cpu** or gpu)
- --n_nodes (76 or **379**)
   + 76 = working memory network, 379 = whole brain network
- --model (**GCNConv**, GCN, or NN)
   + GCNConv = Residual GCN, GCN = Simple GCN, NN = Multi Layer Peceptron
- --seed (default 123)
- --epochs (default 100)
- --lr (default 1e-3)

