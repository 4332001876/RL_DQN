import math
import numpy as np
import csv,pprint
import random
import time
import collections
import copy


import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.animation as animation

PRE_TRAINED=True#是否第一次训练
MODEL_FILENAME="DQN numpy.csv"


MAX_A_P=8.0#人最大加速度
MAX_A_CAR=5.0#车最大加速度
FRICTION_CAR=0.5#车摩擦力产生的加速度与速度之比
FRICTION_P=1.3#人摩擦力产生的加速度与速度之比

CAR_LENGTH=4.8#汽车长
CAR_WIDTH=2.0#汽车宽
TURN_RADIUS=20.0#汽车转弯半径

dt=0.04#欧拉积分时步
STEP_N=7#每积分n次记录一次经验

MAX_EXP_SIZE=100000#经验池大小
DISCOUNT_RATE=0.99#强化学习折扣率
EPSILON=0.1#采取随机加速方向概率,强化学习中的epsilon_greedy策略

BATCH_SIZE=64
EPOCH_SIZE=300








class NN:#neural network
    
    def __init__(self,size):
        self.weight=[]
        self.sum_adjust_weight=[]
        self.layer=[]
        self.bias=[]
        self.sum_adjust_bias=[]
        self.size=size

        self.total_loss=0.0
        self.total_correct=0
        for i in range(len(size)):
            self.layer.append(np.zeros(size[i],dtype=np.float16))
            self.bias.append(np.zeros(size[i],dtype=np.float16))
            self.sum_adjust_bias.append(np.zeros(size[i],dtype=np.float16))
        for i in range(len(size)-1):
            self.weight.append(np.random.rand(size[i],size[i+1])-0.5)#0-1
            self.sum_adjust_weight.append(np.zeros((size[i],size[i+1]),dtype=np.float16))

            

    def forward(self,layer_input):#sigmoid
        self.layer[0]=np.array(layer_input)
        for i in range(1,len(self.layer)):
            self.layer[i]=self.layer[i-1]@self.weight[i-1]
            for j in range(len(self.layer[i])):
                self.layer[i][j]=relu(self.layer[i][j]+self.bias[i][j])
        
        return self.layer[-1]

    def loss_forward(self,layer_input,result_est):
        result=self.forward(layer_input)
        result_est=np.array(result_est)
        loss=0.5*(result_est-result)*(result_est-result)
        return loss

    #原网络减去adjust完成一次梯度下降
    def backward(self,layer_input,result_est):#
        result_est=np.array(result_est)
        adjust=[[] for i in range(len(self.size))]
        result=self.forward(layer_input)
        loss=0.5*(result_est-result)*(result_est-result)
        self.total_loss+=np.average(loss)
        adjust[-1]=(result-result_est)*np.array(list( map(d_relu,result) ))#BP1,map function
        for i in range(len(self.size)-2,-1,-1):
            adjust[i]=  (adjust[i+1]@(self.weight[i].T))  * np.array(list( map(d_relu,self.layer[i]) ) )#BP2
        for i in range(len(self.size)-1,-1,-1):
            self.sum_adjust_bias[i]=self.sum_adjust_bias[i]+adjust[i]   #BP3,sum up in a batch
        for i in range(len(self.weight)-1,-1,-1):
            self.sum_adjust_weight[i]=self.sum_adjust_weight[i]+(self.layer[i].reshape(-1,1))@adjust[i+1].reshape(1,-1)#BP4
        
    def train(self,training_set,learning_rate=0.005,epoch_size=EPOCH_SIZE,batch_size=BATCH_SIZE,times=BATCH_SIZE*EPOCH_SIZE):#training_set:[(input,label), ...]
        #np.random.shuffle(training_set)
        
        
        #for count_epoch in range(int(times/epoch_size/batch_size)):
        #randseries=np.random.randint(0,len(training_set),size=epoch_size*batch_size)
        num=0
        self.total_loss=0.0
        self.total_correct=0.0
        for j in range(epoch_size):
            for i in range(len(self.size)):
                self.sum_adjust_bias[i]=np.zeros(self.size[i],dtype=np.float16)
            for i in range(len(self.size)-1):
                self.sum_adjust_weight[i]=np.zeros((self.size[i],self.size[i+1]),dtype=np.float16)

            for k in range(batch_size):
                
                self.backward(training_set[num][0],training_set[num][1])
                num+=1
                     
            for i in range(len(self.weight)):
                self.weight[i]=self.weight[i]-learning_rate*self.sum_adjust_weight[i]/float(batch_size)
            for i in range(len(self.bias)):
                self.bias[i]=self.bias[i]-learning_rate*self.sum_adjust_bias[i]/float(batch_size)
        loss=self.total_loss/float(epoch_size*batch_size)

        print("ave_loss:%f"%(loss)     )
        
        #learning_rate=learning_rate*0.99
        #if(count_epoch%2==0):
        return loss

            
    def copy_value_from_csv(self,filename,size):
        with open(filename,'r',encoding='utf-8-sig') as infile:
            table=[row for row in csv.reader(infile)]
        count_line=0
        for i in range(len(self.weight)):
            for j in self.weight[i]:
                for k in range(size[i+1]):
                    j[k]=float(table[count_line][k])
                count_line+=1
                
        for i in range(len(self.bias)):
            for j in range(len(self.bias[i])):  
                self.bias[i][j]=float(table[count_line][j])
            count_line+=1
                
    def write_into_csv(self,filename):
        with open(filename, 'w') as file:
            for i in self.weight:
                for j in i:
                    for k in j:
                        file.write("%f,"%k)
                    file.write("\n") 
            for i in self.bias:
                for j in i:
                
                    file.write("%f,"%k)
                file.write("\n") 
def relu(x):
    if x>=0:
        return x
    else:
        return 0.01*x
    
def d_relu(x):
    if x>=0:
        return 1
    else:
        return 0.01
    
    
    
class Env:#强化学习环境
    def __init__(self):
        x_p_x=0.0
        x_p_y=0.0
        while abs(x_p_x)<=0.5* CAR_LENGTH:
            x_p_x=np.random.normal(loc=0,scale=25.0)#离车越近，越需要精细操作，越需要训练
        while abs(x_p_y)<=0.5*CAR_WIDTH:
            x_p_y=np.random.normal(loc=0,scale=25.0)
        self.x_p=np.array([x_p_x,x_p_y],dtype=np.float16)#-100~100
        self.v_p=np.array([0,0],dtype=np.float16)
        self.x_car=np.array([0,0],dtype=np.float16)
        self.v_car=0.0
        self.direct_car=0.0
        self.win_fail=0#1:win  -1:fail
        
        
    def restart(self):
        x_p_x=0.0
        x_p_y=0.0
        while abs(x_p_x)<=0.5* CAR_LENGTH:
            x_p_x=np.random.normal(loc=0,scale=25.0)#离车越近，越需要精细操作，越需要训练
        while abs(x_p_y)<=0.5*CAR_WIDTH:
            x_p_y=np.random.normal(loc=0,scale=25.0)
        self.x_p=np.array([x_p_x,x_p_y],dtype=np.float16)#-100~100
        self.v_p=np.array([0,0],dtype=np.float16)
        self.x_car=np.array([0,0],dtype=np.float16)
        self.v_car=0.0
        self.direct_car=0.0
        self.win_fail=0#1:win  -1:fail
        
    def restart_validation(self):
        self.x_p=np.array([50.0,0.0],dtype=np.float16)#-100~100
        self.v_p=np.array([0,0],dtype=np.float16)
        self.x_car=np.array([0,0],dtype=np.float16)
        self.v_car=0.0
        self.direct_car=0.0
        self.win_fail=0#1:win  -1:fail
        
    def get_anim_state(self):

        return np.hstack((self.x_p,self.x_car,[self.direct_car]))

    
    
    def get_state(self):#获得状态值，可作为神经网络输入
        #以汽车为原点建立自然坐标系，y轴为汽车速度方向，车右边为x轴正方向
        y_direct=[math.cos(self.direct_car),math.sin(self.direct_car)]#汽车前进方向
        x_direct=[math.sin(self.direct_car),-math.cos(self.direct_car)]#汽车右边
        x_p_re=[0,0]
        x_p_re[0]=np.dot(self.x_p-self.x_car,x_direct)#x
        x_p_re[1]=np.dot(self.x_p-self.x_car,y_direct)#y
        
        v_p_re=[0,0]
        v_p_re[0]=np.dot(self.v_p,x_direct)#x
        v_p_re[1]=np.dot(self.v_p,y_direct)#y
        return np.hstack((x_p_re,v_p_re,[self.v_car]))
        #if win_fail!=-1:
        
        #else:
            #return np.hstack(([0.0,0.0],v_p_re,[self.v_car]))
    
    def step(self,action,dt):#欧拉积分
        state=self.get_state()
        #判断输赢
        if abs(state[0])<0.5*CAR_WIDTH and  abs(state[1])<0.5* CAR_LENGTH:
            self.win_fail=-1
        if state[0]**2+state[1]**2>40000:
            self.win_fail=1
        if action<12:
            a_p_direct=self.direct_car+2*math.pi/12.0*action#相对于车的加速方向
            a_p=MAX_A_P*np.array([math.cos(a_p_direct),math.sin(a_p_direct)],dtype=np.float16)
            a_p=a_p-FRICTION_P*self.v_p#考虑摩擦力，其与速度成正比
        else:#第13个动作表示完全不动
            a_p=-FRICTION_P*self.v_p#考虑摩擦力，其与速度成正比
        v_car_direct=np.array([math.cos(self.direct_car),math.sin(self.direct_car)],dtype=np.float16)
        if state[1]>=0:#人在汽车的前方
            a_car=MAX_A_CAR-FRICTION_CAR*self.v_car#汽车向前加速
        else:
            a_car=-MAX_A_CAR-FRICTION_CAR*self.v_car#汽车向后加速
        
        #开始更新状态
        if state[0]>=0:#人在汽车的右方
            self.direct_car-=self.v_car*dt/TURN_RADIUS#汽车向右转弯
        else:
            self.direct_car+=self.v_car*dt/TURN_RADIUS#汽车向左转弯
        
        self.v_car+=a_car*dt
        self.x_car=self.x_car+dt*self.v_car*v_car_direct
        
        self.v_p=self.v_p+a_p*dt
        self.x_p=self.x_p+self.v_p*dt
        
        return self.win_fail
    
def nn_validation(nn,env,val_file,anim_file,random_start=False):#验证神经网络性能
    if random_start:
        env.restart()
    else:
        env.restart_validation()
    trace=[]
    count=0
    while count<1000 and env.win_fail==0:
        state=env.get_state()
        action=np.argmax(nn.forward(state))
        for i in range(STEP_N):
            env.step(action,dt)
        trace.append(env.get_anim_state())#np.hstack((self.x_p,self.x_car,[self.direct_car]))
        count+=1
    trace=np.array(trace)
    
    fig, ax = plt.subplots(figsize=(15, 10),dpi=150)
    plt.title("Trace Validation",fontsize=24)#图标题,设置字体大小   
    plt.plot(trace[:,0],trace[:,1])
    plt.plot(trace[:,2],trace[:,3])

    plt.margins(x=0,y=0)
    plt.tick_params('both', labelsize=16,length=6)#设置数字大小,length是设置坐标轴刻度线长度    
    plt.legend(['man','car'], fontsize=18)#标注legend    
    plt.xlabel('x', fontsize=18)#坐标轴标题   
    plt.ylabel('y', fontsize=18)
    #ax1.set_xticks([20*i for i in range(9)])#设置刻度线打在哪几个数上 
    plt.gca().set_aspect("equal")
    fig.savefig(val_file)
    plt.show()
    
    
    anim(trace,min(200,len(trace)),anim_file) 
    return len(trace)
    
def anim(data,frames,filename,faster=4.0):#输出动画，并保存为文件，faster为动画加速倍数，默认为4倍
    fig, ax = plt.subplots()
    
    
    
    ani = animation.FuncAnimation(fig, update,frames=frames,fargs=(data,ax), interval=int(dt*STEP_N))
    #ani.to_jshtml(fps=int(1/(dt*STEP_N)), embed_frames=True, default_mode=None)
    
    ani.save(filename+'.gif', writer='Pillow',fps=max(1,int(faster/(dt*STEP_N))))
    plt.show()
    
def update(i,data,ax):#更新动画帧的函数，是anim函数的附属函数
    #data[i]=np.hstack((self.x_p,self.x_car,[self.direct_car]))
    x_p=np.array([data[i][0],data[i][1]])
    x_car=np.array([data[i][2],data[i][3]])
    
    direct_car=data[i][4]
    delta_y=0.5*CAR_LENGTH*np.array([math.cos(direct_car),math.sin(direct_car)])#汽车前进方向
    delta_x=0.5*CAR_WIDTH*np.array([math.sin(direct_car),-math.cos(direct_car)])#汽车右边
    line=[]
    line.append(x_car+delta_x+delta_y)
    line.append(x_car-delta_x+delta_y)
    line.append(x_car-delta_x-delta_y)
    line.append(x_car+delta_x-delta_y)
    line.append(line[0])#要围成封闭矩形
    line=np.array(line)
    line_x=line[:,0]
    line_y=line[:,1]
    ax.clear()
    ax.plot(line_x,line_y)
    ax.scatter([x_p[0]],[x_p[1]],s=10)
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    ax.set_aspect('equal')
    return ax
    

def model_result_3D(nn):#可视化网络对各种输入的对应输出，画出3D图像，图像z轴的值是0-12的整数，代表小人动作的编号。
# Load and format data


    x = np.linspace(-30,30, 60)
    y = np.linspace(-30,30, 60)
    x, y = np.meshgrid(x, y)
    z=np.zeros((60,60),dtype=np.float16)
    for i in range(60):
        for j in range(60):
            z[i][j]=np.argmax(nn.forward([x[i][j],y[i][j],0.0,0.0,10.0]))
        
    #region = np.s_[5:50, 5:50]
    #x, y, z = x[region], y[region], z[region]
    
    # Set up plot
    plt.figure(figsize=(27, 18),dpi=150)
    fig, ax = plt.subplots(figsize=(27, 18),dpi=150,subplot_kw=dict(projection='3d'))
    
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    fig.savefig('output 3D.png')
    plt.show()
    
size= [5,12,16,13] 
nn=NN(size)

nn.copy_value_from_csv(MODEL_FILENAME, size)


val_env=Env()#验证训练效果的专用环境 
    
    
for i in range(20):   
    anim_file='performance validation%03d'%(i+1)        
    val_file='trace validation%03d.png'%(i+1)
    nn_validation(nn,val_env,val_file,anim_file,random_start=True)
 



model_result_3D(nn)
