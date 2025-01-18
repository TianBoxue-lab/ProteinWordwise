import os,urllib
from Bio.PDB import PDBParser,DSSP
import pandas as pd

def download_uniprot_AF2_model(id,outname):
    # download uniprot AF2 model
    pdb_url = 'https://alphafold.ebi.ac.uk/files/AF-%s-F1-model_v4.pdb'%id
    req = urllib.request.Request(url=pdb_url)
    response = urllib.request.urlopen(req)
    content = response.read().decode('utf-8')
    with open(outname,'w') as f:
        f.write(content)

def convert_ss8_to_ss3(ss8):
    ss3 = dict.fromkeys(['H','G','I'],'H')
    ss3.update(dict.fromkeys(['-','T','S'],'L'))
    ss3.update(dict.fromkeys(['B','E'],'S'))
    return ''.join(ss3[i] for i in ss8)

def get_ss3(input_name,res_idxs):
    parser = PDBParser()
    structure = parser.get_structure('pdb',input_name)
    model = structure[0]
    dssp = DSSP(model,input_name,dssp='dssp')
    ss8 = ''.join([i[2] for i in dssp if i[0] - 1 in res_idxs])
    return convert_ss8_to_ss3(ss8)

def extract_ss3(uniprot_id,seq,outdir):
    pdbname = os.path.join(outdir,uniprot_id + '_AF2.pdb')
    download_uniprot_AF2_model(uniprot_id, pdbname)
    print('Finished downloading AF2 model: %s'%uniprot_id)
    ss3_list = get_ss3(pdbname,range(len(seq)))
    assert len(ss3_list) == len(seq)
    pd.to_pickle(ss3_list, os.path.join(outdir,uniprot_id + '_AF2_ss3.pkl'))
    print('Finished extracting ss3 from AF2 model')