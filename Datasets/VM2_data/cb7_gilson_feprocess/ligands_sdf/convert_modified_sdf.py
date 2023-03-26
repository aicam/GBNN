import sys
from verachem.files import fileReaders

def convertFromModifiedMol(lines):
        
    # Fix the counts line   
    countsLine = lines[3]
    numAtoms = int(countsLine[:5])
    numBonds = int(countsLine[5:10])
    if numAtoms > 999 or numBonds > 999:           
            print "Molecule:"
            print lines[0]
            print "Too many atoms or bonds to convert. atoms: %d, bonds: %d" % (numAtoms, numBonds) 
            sys.exit()
 
    lines[3] = '%3d%3d' % (numAtoms,numBonds) + countsLine[10:]
        
    # Fix the bonds block
    firstBondLineIndex = numAtoms + 4       
    for i in range(firstBondLineIndex, firstBondLineIndex + numBonds):
        atom1 = int(lines[i][:5])
        atom2 = int(lines[i][5:10])
        lines[i] = '%3d%3d' % (atom1, atom2) + lines[i][10:]            
          
    return lines       

inputFilename = sys.argv[1]
outputFilename = sys.argv[2]
output = open(outputFilename,'w')

sdReader = fileReaders.SdFileReaderLight(filename = inputFilename,
                                         modifiedMOL = True)

while 1:
    molReader = sdReader.getNextMolFile()
    if molReader == None:
        break

    curLines = sdReader.getCurLines()
    newLines = convertFromModifiedMol(curLines)

    for line in newLines:
        output.write(line)
    output.write('$$$$\n')
    
