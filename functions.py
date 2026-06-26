#Write a function in python to add two numbers
def add(a, b):
   return a + b
a = 2
b = 3
result = add(a, b)
print("The sum is:", result)

#Check whether the number is positive,zero or negative
num = -5
if num > 0:
   print("Positive")
elif num == 0:
   print("Zero")
else:
   print("Negative")    

#Find factorial of a number using recursion
def factorial(n):
   
   if n == 0 or n == 1:
       return 1
   
   return n * factorial(n - 1)

num = 5
print(f"The factorial of {num} is {factorial(num)}")

#Display all natural numbers from 1 to 10
num = 1
print("The natural numbers from 1 to 10 are:")
while num <= 10:
   print(num)
   num = num + 1
print("Program completed successfully.")   

 

   
   