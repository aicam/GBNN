class GBNSR6_Error(Exception):
    def __init__(self, msg = "GBNSR6 python error"):
        self.msg = msg
    def __str__(self):
        return self.msg

class PositionException(GBNSR6_Error):
    ''' position is not convertable '''
    pass