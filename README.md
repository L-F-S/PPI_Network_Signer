# SIGNAL
## SIGN Annotation aLgorithm
Assign sign (+ or -) to directed edges of protein-protein interaction networks

TODO: in un futuro lontano che renderai la repo pubblica, controlla che utils/extract_kegg_interactions.py funga dentro sta repo


tuttto il data preproc in realta/ lo dovresti eliminare e  o mettere in un altra repo che si chiama come downloaddare un sacco di datasets bvelocemente
in cui metti anche : depod_scraping.py

# Pipeline:
## Preprocessing:
-the data should look like blabla
## Inputs:
  shape of inputs:
  tran=ining data:
  .lbl.tsv
  .w8.tsv

'python SIGNAL_ft_gen_iterative.py -h' for input help


# Usage:
step 1 feature creation:
ci sono due modi di usare SIGNAL_ft_gen: 
  1 chiamare gli input da linea di compando: python sugnal_ft_gen -a -b- c
  2 importare le funzioni in un altro file (come dentro SIGNAL_pipeline dentro utility scripts) vederlo come esempio
