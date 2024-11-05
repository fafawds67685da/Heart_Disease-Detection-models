import matplotlib.pyplot as m
import numpy as n
import pandas as p

num_cases = 100

#1 ( bada wala bp)
bp_systolic = n.random.randint(90, 180, num_cases)
#2 ( chhota wala bp)
bp_diastolic = n.random.randint(60, 120, num_cases)
#3 (pulse)
heart_rate = n.random.randint(50, 120, num_cases)
#4 (cholestrol level)
cholesterol_level = n.random.randint(150, 300, num_cases)
#5 (glucose level)
glucose_level = n.random.randint(70, 200, num_cases)
#6 (BMI)
BMI = n.round(n.random.uniform(18, 40, num_cases), 2)
#7 (physical activity)
physical_activity = n.random.randint(1, 11, num_cases)
#8 Oxygen levels
oxygen_saturation = n.random.randint(90, 100, num_cases)

# 9 Physical Symptom ( chest pain , shortness of breath , palpitations)
#       Enter 1 for chest pain
#       Enter 2 for shortness of breath
#       Enter 3 for palpitations
#       Enter 4 for chest pain and shortness of breath
#       Enter 5 for shortness of breath and palpitations
#       Enter 6 for chest pain and palpitations
#       Enter 7 for all of them

symptoms = n.random.randint(1, 8, num_cases)

# 10 result probqability
heart_complexity_probability = n.round(n.random.uniform(0, 100, num_cases), 2)



bp1=(bp_systolic - n.min(bp_systolic)) / (n.max(bp_systolic) - n.min(bp_systolic))
bp2=(bp_diastolic - n.min(bp_diastolic)) / (n.max(bp_diastolic) - n.min(bp_diastolic))
hr=(heart_rate - n.min(heart_rate)) / (n.max(heart_rate) - n.min(heart_rate))
col=(cholesterol_level - n.min(cholesterol_level)) / (n.max(cholesterol_level) - n.min(cholesterol_level))

gl=(glucose_level - n.min(glucose_level)) / (n.max(glucose_level) - n.min(glucose_level))
bmi=(BMI - n.min(BMI)) / (n.max(BMI) - n.min(BMI))
pa=(physical_activity - n.min(physical_activity)) / (n.max(physical_activity) - n.min(physical_activity))
os=(oxygen_saturation - n.min(oxygen_saturation)) / (n.max(oxygen_saturation) - n.min(oxygen_saturation))
s=(symptoms - n.min(symptoms)) / (n.max(symptoms) - n.min(symptoms))
y=(heart_complexity_probability - n.min(heart_complexity_probability)) / (n.max(heart_complexity_probability) - n.min(heart_complexity_probability))



w1=0
w2=0
w3=0

w4=0
w5=0
w6=0

w7=0
w8=0
w9=0

b=0
lr=0.01



def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,w1,w2,w3,w4,w5,w6,w7,w8,w9,b):
  z1= x1*w1
  z2= x2*w2
  z3= x3*w3

  z4= x4*w4
  z5= x5*w5
  z6= x6*w6

  z7= x7*w7
  z8= x8*w8
  z9= x9*w9



  return (z1+z2+z3+z4+z5+z6+z7+z8+z9+b)

def cost_func(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,y):
  y2=predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)
  mse=(y2-y)**2
  return n.mean(mse)




def update(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,lr):

  dLdW=n.mean(-2*bp1*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW2=n.mean(-2*bp2*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW3=n.mean(-2*hr*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))

  dLdW4=n.mean(-2*col*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW5=n.mean(-2*gl*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW6=n.mean(-2*bmi*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))

  dLdW7=n.mean(-2*pa*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW8=n.mean(-2*os*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))
  dLdW9=n.mean(-2*s*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))

  dLdb=n.mean(-2*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b)))

  w1=w1-lr*dLdW
  w2=w2-lr*dLdW2
  w3=w3-lr*dLdW3

  w4=w4-lr*dLdW4
  w5=w5-lr*dLdW5
  w6=w6-lr*dLdW6

  w7=w7-lr*dLdW7
  w8=w8-lr*dLdW8
  w9=w9-lr*dLdW9
  b=b-lr*dLdb
  return w1,w2,w3,w4,w5,w6,w7,w8,w9,b


def train(bp1,bp2,hr,col,gl,bmi,pa,os,s,y,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,lr,tol=1e-13,n_epochs=100, verbose=False):
  weights1=[w1]
  weights2=[w2]
  weights3=[w3]

  weights4=[w4]
  weights5=[w5]
  weights6=[w6]

  weights7=[w7]
  weights8=[w8]
  weights9=[w9]

  biases=[b]
  costs=[]

  ct=1
  while True:
    cost=cost_func(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,y)
    costs.append(cost)
    if len(costs)>1 and abs(costs[-2]-costs[-1]) < tol:
      break



    w1,w2,w3,w4,w5,w6,w7,w8,w9,b=update(bp1,bp2,hr,col,gl,bmi,pa,os,s,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,lr)

    weights1.append(w1)
    weights2.append(w2)
    weights3.append(w3)

    weights4.append(w4)
    weights5.append(w5)
    weights6.append(w6)

    weights7.append(w7)
    weights8.append(w8)
    weights9.append(w9)

    biases.append(b)


  return weights1,weights2,weights3,weights4,weights5,weights6,weights7,weights8,weights9,biases,costs

w1,w2,w3,w4,w5,w6,w7,w8,w9,b,c=train(bp1,bp2,hr,col,gl,bmi,pa,os,s,y,w1,w2,w3,w4,w5,w6,w7,w8,w9,b,lr, verbose=True)




bp_systolic2 = []
bp_diastolic2 = []
heart_rate2 = []
cholesterol_level2 = []
glucose_level2 = []
BMI2 = []
physical_activity2 = []
oxygen_saturation2 = []
symptoms2 = []
heart_complexity_probability2 = []
num_cases2 = int(input("Enter number of patients\n"))

for i in range(1,6,1):
  a1=int(input(""))  

#1 ( bada wala bp)
bp_systolic2 = n.random.randint(90, 180, num_cases2)
#2 ( chhota wala bp)
bp_diastolic2 = n.random.randint(60, 120, num_cases2)
#3 (pulse)
heart_rate2 = n.random.randint(50, 120, num_cases2)
#4 (cholestrol level)
cholesterol_level2 = n.random.randint(150, 300, num_cases2)
#5 (glucose level)
glucose_level2 = n.random.randint(70, 200, num_cases2)
#6 (BMI)
BMI2 = n.round(n.random.uniform(18, 40, num_cases2), 2)
#7 (physical activity)
physical_activity2 = n.random.randint(1, 11, num_cases2)
#8 Oxygen levels
oxygen_saturation2 = n.random.randint(90, 100, num_cases2)

# 9 Physical Symptom ( chest pain , shortness of breath , palpitations)
#       Enter 1 for chest pain
#       Enter 2 for shortness of breath
#       Enter 3 for palpitations
#       Enter 4 for chest pain and shortness of breath
#       Enter 5 for shortness of breath and palpitations
#       Enter 6 for chest pain and palpitations
#       Enter 7 for all of them

symptoms2 = n.random.randint(1, 8, num_cases2)

# 10 result probqability
heart_complexity_probability2 = n.round(n.random.uniform(0, 100, num_cases2), 2)


bp12=(bp_systolic2 - n.min(bp_systolic2)) / (n.max(bp_systolic2) - n.min(bp_systolic2))
bp22=(bp_diastolic2 - n.min(bp_diastolic2)) / (n.max(bp_diastolic2) - n.min(bp_diastolic2))
hr2=(heart_rate2 - n.min(heart_rate2)) / (n.max(heart_rate2) - n.min(heart_rate2))
col2=(cholesterol_level2 - n.min(cholesterol_level2)) / (n.max(cholesterol_level2) - n.min(cholesterol_level2))

gl2=(glucose_level2 - n.min(glucose_level2)) / (n.max(glucose_level2) - n.min(glucose_level2))
bmi2=(BMI2 - n.min(BMI2)) / (n.max(BMI2) - n.min(BMI2))
pa2=(physical_activity2 - n.min(physical_activity2)) / (n.max(physical_activity2) - n.min(physical_activity2))
os2=(oxygen_saturation2 - n.min(oxygen_saturation2)) / (n.max(oxygen_saturation2) - n.min(oxygen_saturation2))
s2=(symptoms2 - n.min(symptoms2)) / (n.max(symptoms2) - n.min(symptoms2))
y2=(heart_complexity_probability2 - n.min(heart_complexity_probability2)) / (n.max(heart_complexity_probability2) - n.min(heart_complexity_probability2))

y_res=predict(bp12,bp22,hr2,col2,gl2,bmi2,pa2,os2,s2,w1[-1],w2[-1],w3[-1],w4[-1],w5[-1],w6[-1],w7[-1],w8[-1],w9[-1],b[-1])



mse=n.mean((y_res-y2)**2)
print(y_res *100)

