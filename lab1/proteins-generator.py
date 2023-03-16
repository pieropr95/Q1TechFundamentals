import numpy as np
import random
import string
import sys

letters = np.array(list(chr(ord('A') + i) for i in range(4)))
#print(letters)


f=open('proteins.csv', 'w+')

f.write("structureId,sequence\n")

print(sys.argv[1])

linesno=int(sys.argv[1])
count = 0

for i in range (linesno):
   chars = ''.join([random.choice(letters) for j in range(random.randrange(1, 256, 2))])
   f.write(str(i+1))
   f.write(",") 
   if (random.randrange(1, 256, 1) ==   i % 512):
      count =  count +1
      f.write("ABCD")
      if (count % 256 ==   0):
         f.write("ABCD")
   f.write(chars)
   f.write("\n")



