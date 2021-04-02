from biopandas.pdb import PandasPdb



f = open('fake_o')
a = f.readlines()
pandaspdb = PandasPdb()
ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=a)
print(ppdb_df.df['OTHERS'].iloc[20])
