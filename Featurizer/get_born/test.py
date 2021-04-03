from biopandas.pdb import PandasPdb



f = open('fake_o')
B = []
for l in f:
    if l[:4] != "rinv":
        continue
    B.append(float(list(filter(None, l.split(' ')))[2].replace('\n', '')))
print(B)

# for l in f:
#     B = []
#     if l[:4] != "rinv":
#         continue
#     i = 0
#     for j in range(len(l)):
#         if i == 0:
#             if l[j] == ' ':
#                 i += 1
#             else:
#                 continue
#         if i == 1:
#             if l[j] == ' ':
#                 continue
#             else:
#                 i += 1
#         if i == 2:
#             if l[j] == ' ':
#                 continue
#             elif


