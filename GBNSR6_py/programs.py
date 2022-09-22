from exceptions import AMBERNOTFound
def find_program(program, search_path = False):
   """ Searches for a program in $AMBERHOME first, then PATH if we allow
       PATH searching.
   """
   import os

   def is_exe(filename):
      return os.path.exists(filename) and os.access(filename, os.X_OK)

   def get_amberhome():
      ambhome = os.getenv('AMBERHOME')
      if ambhome == None:
         raise AMBERNOTFound('AMBERHOME is not set!')
      return ambhome

   # Check to see that a path was provided in the program name
   fpath, fname = os.path.split(program)
   if fpath:
      if is_exe(program): return program

   # If not (and it generally isn't), then look in AMBERHOME
   amberhome = get_amberhome()

   if is_exe(os.path.join(amberhome, 'bin', program)):
      return os.path.join(amberhome, 'bin', program)

   # If we can search the path, look through it
   if search_path:
      for path in os.environ["PATH"].split(os.pathsep):
         exe_file = os.path.join(path, program)
         if is_exe(exe_file):
            return exe_file # if it's executable, return the file

   return None  # if program can still not be found... return None
