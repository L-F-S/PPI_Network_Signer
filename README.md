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
  - step 1 feature creation:
    
       SIGNAL_ft_gen_iterative.py can be called in two main ways to generate features:
      1- from the command line 
      `python SIGNAL_ft_gen_iterative.py -s S_cerevisiae -e edges_file -n network_file -p perturbations_file`
      2- import functions in a different python script by:
      'from SIGNAL_ft_gen_iterative import generate_similarity_matrix_wrapper, generate_features_different_knockouts_iterative`
      for multi-threading option, use SIGNAL_ft_gen_parallel.py and import generate_features_different_knockouts function
