import tpr_fpr_auc
from mpi4py import MPI

number_of_windowsizes = len(tpr_fpr_auc.windowsizes)

world_comm = MPI.COMM_WORLD

job_number = world_comm.Get_rank()

i_ = int(job_number%number_of_windowsizes)
j_ = int(job_number/number_of_windowsizes)

tpr_fpr_auc.get_tpr_fpr_auc(i_,j_)