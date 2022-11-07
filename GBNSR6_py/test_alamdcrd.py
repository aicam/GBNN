from alamdcrd import MutantMdcrd
traj = 'prods/prod1.crd'
org_prm = 'prods/6m0j.prmtop'
mut_prm = 'prods/complex.prmtop'
com_mut = MutantMdcrd(traj, org_prm, mut_prm)
com_mut.MutateTraj('mutant_complex.crd')