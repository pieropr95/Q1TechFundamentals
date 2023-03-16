import numpy as np
import random
import string
import sys



f=open('computers.csv', 'w+')

f.write("id,price,speed,hd,ram,screen,cores,cd,laptop,trend\n")

linesno=int(sys.argv[1])
count = 0

for i in range (linesno):

#1,3116;25;80;4;14;no;no;yes;94;2
   price=random.randrange(500, 4350, 1) 
   if (price < 3000):
         speed=random.randrange(15, 25, 1)
   else:
         speed=random.randrange(30, 40, 1)
   if (price < 3000):
         hd=random.randrange(8, 32, 4)
   else:
         hd=random.randrange(1, 8, 2)
   if (price < 3000):
         ram=random.randrange(2, 16, 4)
   else:
         ram=random.randrange(16, 64, 4)
   if (price < 1200):
         screen=random.randrange(11, 14, 1)
   else:
         screen=random.randrange(15, 25, 2)
   if (price < 2000):
         cores=random.randrange(4, 12, 2)
   else:
         cores=random.randrange(12, 32, 2)
   if (price > 2000 and cores > 16):
         trend = random.randrange(4, 6, 1)
   else:
         trend = random.randrange(1, 3, 1)
   if (price % 4 == 0):
         cd = "yes"
   else:
         cd = "no"
   if (screen < 15 and cd=="no"):
         laptop = "yes"
   else:
         laptop = "no"
   f.write(str(i+1))
   f.write(",") 
   f.write(str(price))
   f.write(",") 
   f.write(str(speed))
   f.write(",") 
   f.write(str(hd))
   f.write(",") 
   f.write(str(ram))
   f.write(",") 
   f.write(str(screen))
   f.write(",") 
   f.write(str(cores))
   f.write(",") 
   f.write(cd)
   f.write(",") 
   f.write(laptop)
   f.write(",") 
   f.write(str(trend))
   f.write("\n")



