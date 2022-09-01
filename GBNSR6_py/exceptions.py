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
