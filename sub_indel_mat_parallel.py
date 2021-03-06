import argparse as ag
import pandas as pd
import numpy as np
import pysam
from mpi4py import MPI

def sub_indel_mat_parallel(bam_path, ref_path, out_path, alphabet=['A', 'C', 'G', 'T', '_']):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    sub_mat = pd.DataFrame(np.zeros((len(alphabet), len(alphabet)), dtype='int'), index=alphabet, columns=alphabet)
    with pysam.AlignmentFile(bam_path, 'rb') as alignment, pysam.Fastafile(ref_path) as ref:
        for ref_contig in alignment.references:
            ref_str = ref.fetch(ref_contig)
            for i, read in enumerate(alignment.fetch(ref_contig)):
                if (i%comm.size == comm.rank):
                    start_flag = False
                    for qi, ri in read.get_aligned_pairs():
                        if (qi != None):
                            if (qi == read.query_alignment_start): start_flag = True 
                        if (start_flag) and (read.seq != None):
                            if (qi == None) and (ref_str[ri].upper() in alphabet):
                                sub_mat.loc['_', ref_str[ri].upper()] += 1
                            elif (ri == None) and (read.seq[qi].upper() in alphabet):  
                                sub_mat.loc[read.seq[qi], '_'] += 1
                            elif (read.seq[qi] != ref_str[ri]) and (read.seq[qi] in alphabet)\
                                  and (ref_str[ri].upper() in alphabet):
                                sub_mat.loc[read.seq[qi], ref_str[ri].upper()] +=1
                               
    comm.Barrier()
    
    if comm.rank == 0:
        fin_mat = pd.DataFrame(np.zeros((len(alphabet), len(alphabet))), index=alphabet, columns=alphabet)
        vals = np.zeros(sub_mat.size, dtype='int')
    else:
        fin_mat = None
        vals = None
        
    comm.Reduce([sub_mat.values.flatten(), MPI.INT],
                [vals, MPI.INT],
                op=MPI.SUM,
                root=0)
    
    if comm.rank == 0:
        fin_mat += vals.reshape(sub_mat.shape)
        fin_mat.to_csv(out_path)

def getArgs():
    parser = ag.ArgumentParser()

    parser.add_argument('-b', '--bam-file',required=True, dest='bam',type=str, help='Sorted and indexed .bam file')
    parser.add_argument('-ref', '--reference-file',required=True,dest='ref',type=str, help='Reference genome in .fasta')
    parser.add_argument('-o','--output-file', required=True, dest='out',type=str, help='Output .csv file')
    
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = getArgs()
    sub_indel_mat_parallel(args['bam'], args['ref'], args['out'])
    