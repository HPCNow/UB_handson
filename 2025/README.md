# Here you have a summary of the commands needed
## clone the repo
git clone https://github.com/HPCNow/UB_handson.git

## access the cluster
ssh ubil3_29@pirineus3.csuc.cat -p 2122

#say yes if asked and insert your password (you will not see the typing)
exit
##copy the file via scp
scp -rp -P 2122 UB_handson ubil3_29@pirineus3.csuc.cat:/home/ubil3_29

## ssh again
ssh ubil3_29@pirineus3.csuc.cat -p 2122

#inside pirineus3
cd UB_handson/2025
module load conda
conda env create -f test-mpi.yml -y ## it has to be created only once

#send a job
cd single
sbatch single_core.slm

#check the status
squeue -u $USER
~                  
