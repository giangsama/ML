import numpy as np
import matplotlib.pyplot as plt

#height
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
#weight
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T 


#building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

#Công thức tính điểm tối ưu trong bài toán linear regression
#       w = A(t).b
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T,y)
w = np.dot(np.linalg.pinv(A),b) #giả nghịch đảo của một ma trận A trong Python sẽ được tính bằng numpy.linalg.pinv(A), pinv là từ viết tắt của pseudo inverse.
print("w = ",w)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145,185,2)
y0 = w_0 + w_1*x0

#Visualize data
plt.plot(X.T,y.T,"ro")
plt.plot(x0,y0,"b-")
plt.axis([140,190,40,75]) # giới hạn trục số cho biểu đồ
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()
