class GBNSR6_Error(Exception):
    def __init__(self, msg = "GBNSR6 python error"):
        self.msg = msg
    def __str__(self):
        return self.msg
class GBNSR6_Warning(Warning):
   """ Base MMPBSA warning class """
   def __init__(self, msg='GBNSR6 warning'):
      self.msg = msg
   def __str__(self):
      return self.msg
class PoistionsNumberWarning(GBNSR6_Warning):
    ''' number of positions in the file are not as expected '''
    pass
class PositionException(GBNSR6_Error):
    ''' position is not convertable '''
    pass

class PositionDecimalExceed(GBNSR6_Error):
    ''' each position can have up to 7 floating decimals and 4 non-floating decimals '''
    pass

class FilePermission(GBNSR6_Error):
    ''' file permission not granted '''
    pass

class LEaPProgramFailed(GBNSR6_Error):
    ''' in case of error in running tleap '''
    pass

class AMBERNOTFound(GBNSR6_Error):
    ''' $AMBERHOME is not defined '''
    pass

class MMPBSA_Error(Exception):
   """ Base MMPBSA error class """
   def __init__(self, msg='MMPBSA error'):
      self.msg = msg
   def __str__(self):
      return self.msg

class TrajError(MMPBSA_Error):
   """ Error in trajectory processing """
   pass

class InternalError(MMPBSA_Error):
   """ Error from buggy coding """
   pass

class MutateError(MMPBSA_Error):
   """ Error mutating a trajectory in alanine scanning """
   pass

class MutantResError(MutateError):
   """ Error if we're mutating to a bad residue """
   pass