from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

# ======================================================================================
# Data preprocessing - everything happens in files a1.py, a2.py, etc.


# ======================================================================================
# Task A1
# Gender detection: male or female

model_A1 = A1()
acc_A1_train = model_A1.train()
acc_A1_test = model_A1.test()
del model_A1


# ======================================================================================
# Task A2
# Emotion detection: smiling or not smiling
model_A2 = A2()
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()


# ======================================================================================
# Task B1
# Face shape recognition: 5 types of face shapes
model_B1 = B1()
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()


# ======================================================================================
# Task B2
# Eye color recognition: 5 types of eye colors
model_B2 = B2()
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()


# ======================================================================================
## Print out your results with following format:
print(
    "TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};".format(
        acc_A1_train,
        acc_A1_test,
        acc_A2_train,
        acc_A2_test,
        acc_B1_train,
        acc_B1_test,
        acc_B2_train,
        acc_B2_test,
    )
)
