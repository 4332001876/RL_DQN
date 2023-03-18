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

PRE_TRAINED=False#有无预训练好的网络
MODEL_FILENAME="DQN numpy.csv"#保存已经训练好的模型的文件名


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
EPSILON=0.02#采取随机加速方向的概率,强化学习中的epsilon_greedy策略

BATCH_SIZE=64#神经网络训练数据集是分成一个个batch给神经网络训练的，此参数即为batch大小
EPOCH_SIZE=300#神经网络一个epoch训练几个batch
EPOCHS=50#这代表神经网络将训练EPOCHS个epoch，这直接与训练时间正相关

class NN:#neural network
    
    def __init__(self,size):
        #搭建神经网络
        self.weight=[]#神经网络权重
        self.sum_adjust_weight=[]#神经网络反向传播中对权重求导所得导数
        self.layer=[]#神经网络输入层、各隐含层和输出层
        self.bias=[]#神经网络各层偏置
        self.sum_adjust_bias=[]#神经网络反向传播中对偏置求导所得导数
        self.size=size#神经网络各层神经元数量

        self.total_loss=0.0#用来计算神经网络损失函数
        #self.total_correct=0
        for i in range(len(size)):
            self.layer.append(np.zeros(size[i],dtype=np.float16))
            self.bias.append(np.zeros(size[i],dtype=np.float16))
            self.sum_adjust_bias.append(np.zeros(size[i],dtype=np.float16))
        for i in range(len(size)-1):
            self.weight.append(np.random.rand(size[i],size[i+1])-0.5)#随机初始化神经网络参数在-0.5~0.5之间
            self.sum_adjust_weight.append(np.zeros((size[i],size[i+1]),dtype=np.float16))

            

    def forward(self,layer_input):#由输入求神经网络输出结果
        self.layer[0]=np.array(layer_input)
        for i in range(1,len(self.layer)):
            self.layer[i]=self.layer[i-1]@self.weight[i-1]
            for j in range(len(self.layer[i])):
                self.layer[i][j]=relu(self.layer[i][j]+self.bias[i][j])
        
        return self.layer[-1]

    def loss_forward(self,layer_input,result_est):#求神经网络损失
        result=self.forward(layer_input)
        result_est=np.array(result_est)
        loss=0.5*(result_est-result)*(result_est-result)
        return loss

    #原网络减去adjust完成一次梯度下降
    def backward(self,layer_input,result_est):#反向传播求神经网络各参数导数，神经网络最重要的算法
        result_est=np.array(result_est)
        adjust=[[] for i in range(len(self.size))]
        result=self.forward(layer_input)
        loss=0.5*(result_est-result)*(result_est-result)
        self.total_loss+=np.average(loss)
        #反向传播四公式
        adjust[-1]=(result-result_est)*np.array(list( map(d_relu,result) ))#BP1,map function
        for i in range(len(self.size)-2,-1,-1):
            adjust[i]=  (adjust[i+1]@(self.weight[i].T))  * np.array(list( map(d_relu,self.layer[i]) ) )#BP2
        for i in range(len(self.size)-1,-1,-1):
            self.sum_adjust_bias[i]=self.sum_adjust_bias[i]+adjust[i]   #BP3,sum up in a batch
        for i in range(len(self.weight)-1,-1,-1):
            self.sum_adjust_weight[i]=self.sum_adjust_weight[i]+(self.layer[i].reshape(-1,1))@adjust[i+1].reshape(1,-1)#BP4
        
    def train(self,training_set,learning_rate=0.001,epoch_size=EPOCH_SIZE,batch_size=BATCH_SIZE,times=BATCH_SIZE*EPOCH_SIZE):#输入训练集开始训练training_set:[(input,label), ...]
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
                
                self.backward(training_set[num][0],training_set[num][1])#反向传播
                num+=1
                     
            for i in range(len(self.weight)):
                self.weight[i]=self.weight[i]-learning_rate*self.sum_adjust_weight[i]/float(batch_size)#调整网络参数
            for i in range(len(self.bias)):
                self.bias[i]=self.bias[i]-learning_rate*self.sum_adjust_bias[i]/float(batch_size)#调整网络参数
        loss=self.total_loss/float(epoch_size*batch_size)

        print("ave_loss:%f"%(loss)     )#输出训练损失
        
        #learning_rate=learning_rate*0.99
        #if(count_epoch%2==0):
        return loss

            
    def copy_value_from_csv(self,filename,size):#从csv中将之前的神经网络参数拷贝进来
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
                
    def write_into_csv(self,filename):#保存现在的神经网络参数
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
def relu(x):#神经网络激活函数ReLU用来增加神经网络的非线性性
    if x>=0:
        return x
    else:
        return 0.01*x
    
def d_relu(x):#神经网络激活函数ReLU的导数
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
        self.x_p=np.array([x_p_x,x_p_y],dtype=np.float16)##开局点坐标在-75~75，呈正态分布
        self.v_p=np.array([0,0],dtype=np.float16)#初始化人的速度为0
        self.x_car=np.array([0,0],dtype=np.float16)#可不失一般性地将小车放在原点
        self.v_car=0.0#初始化车的速度为0
        self.direct_car=0.0#可不失一般性地将小车方向设为向右，x轴正方向
        self.win_fail=0#1:win  -1:fail
        
        
    def restart(self):#随机生成开局
        x_p_x=0.0
        x_p_y=0.0
        while abs(x_p_x)<=0.5* CAR_LENGTH:
            x_p_x=np.random.normal(loc=0,scale=25.0)#离车越近，越需要精细操作，越需要训练
        while abs(x_p_y)<=0.5*CAR_WIDTH:
            x_p_y=np.random.normal(loc=0,scale=25.0)
        self.x_p=np.array([x_p_x,x_p_y],dtype=np.float16)#开局点坐标在-75~75，呈正态分布
        self.v_p=np.array([0,0],dtype=np.float16)#初始化人的速度为0
        self.x_car=np.array([0,0],dtype=np.float16)#可不失一般性地将小车放在原点
        self.v_car=0.0#初始化车的速度为0
        self.direct_car=0.0#可不失一般性地将小车方向设为向右，x轴正方向
        self.win_fail=0#1:win  -1:fail
        
    def restart_validation(self):#生成固定的开局，专门用于验证神经网络性能
        self.x_p=np.array([50.0,0.0],dtype=np.float16)#-100~100
        self.v_p=np.array([0,0],dtype=np.float16)
        self.x_car=np.array([0,0],dtype=np.float16)#可不失一般性地将小车放在原点
        self.v_car=0.0
        self.direct_car=0.0#可不失一般性地将小车方向设为向右，x轴正方向
        self.win_fail=0#1:win  -1:fail
        
    def get_anim_state(self):#获得输出动画文件所需要的数据

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
    
    def step(self,action,dt):#欧拉积分，通过泰勒展开一阶近似，从当前状态和动作输入推出下一状态情况
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
    


class ExpBuffer:#经验池
    def __init__(self):    
        self.exp=collections.deque([],maxlen=MAX_EXP_SIZE)#双端队列，元素为字典exp:{"state":,"action":,"reward":,"state_next":,"is_done":}(即当前状态、动作、奖励、下一状态、博弈是否结束)
        self.anim=collections.deque([],maxlen=MAX_EXP_SIZE)#双端队列，存动画素材
        
    def add(self,exp,frame_data):#添加元素时尽量使用这个函数使得exp和anim可以同步添加
        self.exp.append(exp)
        self.anim.append(frame_data)
        
        
    def __len__(self):#使python内置的len()函数可以获取经验池长度
        return len(self.exp)
    
    def get_sample(self,size):#获取经验池样本
        randseries=np.random.randint(0,len(self.exp),size=size)#先批量生成好随机数以提高效率
        sample=[]
        for i in range(size):
            sample.append(self.exp[randseries[i]])
        return sample
    
    def get_anim_data(self,start_point):#从anim的第start_point个数据点开始获取动画数据，直到此轮博弈终止的那个数据点
        anim_data=[]
        i=start_point
        while i<len(self.exp) and self.exp[i]["is_done"]==False:#逻辑短路
            anim_data.append(self.anim[i])
            i+=1
        return anim_data
    
            
def rand_exp_buffer(env,buffer,size):#随机构造经验池
    count=0
    env.restart()
    state=env.get_state()
    randseries=np.random.randint(0,13,size=int(size/50)+1)#先批量生成好随机数以提高效率
    flag_done=False
    while count<size:  
        if env.win_fail==0:
            anim_data=env.get_anim_state()
            action=randseries[int(count/50)]#随机选取动作
            for i in range(STEP_N):
                env.step(action,dt)
                if env.win_fail!=0:
                    flag_done=True
                
                    
            new_state=env.get_state()
            reward=0.5*math.sqrt(new_state[0]**2+new_state[1]**2)+(math.sqrt(new_state[0]**2+new_state[1]**2)-math.sqrt(state[0]**2+state[1]**2))/dt/STEP_N#PD控制，以离小车距离和远离小车速度的线性组合作为奖励函数
            exp={"state":state,"action":action,"reward":reward,"state_next":new_state,"is_done":flag_done}#从当前数据建立经验字典
            
            buffer.add(exp,anim_data)#将经验装入经验池
            state=new_state
            count+=1
            
        else:   
            env.restart()
            flag_done=False
            state=env.get_state()

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
    ax.set_xlim(-160,160)
    ax.set_ylim(-160,160)
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


def nn_validation(nn,env,val_file,anim_file,random_start=False):#验证神经网络性能，画出小车与人追逃的轨迹图以及动画，并返回小人坚持不被小车追上的时长作为神经网络表现分
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
    

def nn_exp_buffer(target_nn,env,buffer,size):#按照目标神经网络的价值判断来构造经验
    count=0
    env.restart()
    state=env.get_state()
    flag_done=False
    while count<size:  
        if env.win_fail==0:
            anim_data=env.get_anim_state()
            #以1-EPSILON的概率按照估计值最大的策略执行动作
            if random.random()>EPSILON:#epsilon_greedy
                action=np.argmax(target_nn.forward(state))
            else:
                action=random.randint(0, 12)#0-12!
            for i in range(STEP_N):
                env.step(action,dt)
                if env.win_fail!=0:
                    flag_done=True
                
                    
            new_state=env.get_state()
            reward=0.5*math.sqrt(new_state[0]**2+new_state[1]**2)+(math.sqrt(new_state[0]**2+new_state[1]**2)-math.sqrt(state[0]**2+state[1]**2))/dt/STEP_N#PD控制，以离小车距离和远离小车速度的线性组合作为奖励函数
            exp={"state":state,"action":action,"reward":reward,"state_next":new_state,"is_done":flag_done}#经验字典，用来作为神经网络输入
            
            buffer.add(exp,anim_data)#将经验装入经验池
            state=new_state
            count+=1
            
        else:   
            env.restart()#如果一次博弈结束，重新设定环境为初始值再开始新一轮博弈
            flag_done=False
            state=env.get_state()

size= [5,12,16,13] #神经网络每层神经元数量
nn=NN(size)#建立神经网络
if PRE_TRAINED:
    nn.copy_value_from_csv(MODEL_FILENAME, size)
target_nn = copy.deepcopy(nn)#建立一个参数暂时相同的目标网络，每隔一定步数更新此网络，以保证训练稳定

env=Env()
buffer=ExpBuffer()
if PRE_TRAINED==False:
    rand_exp_buffer(env,buffer,10000)#只有第一次训练才要，随机生成训练标签

val_env=Env()#验证训练效果的专用环境

loss=[]#损失函数，在强化学习中可理解为神经网络稳定性
pts=[]#得分，代表小人坚持不被小车撞的时长

for i in range(EPOCHS):#训练EPOCHS个epoch
    training_set=[]
    nn_exp_buffer(target_nn,env,buffer,10000) #构造经验装入经验池
    sample=buffer.get_sample(BATCH_SIZE*EPOCH_SIZE)
    for j in range(BATCH_SIZE*EPOCH_SIZE):
        state=sample[j]['state']
        action=sample[j]['action']
        reward=sample[j]['reward']
        if sample[j]["is_done"]==False:
            next_state=sample[j]["state_next"]
            Qvalue=target_nn.forward(next_state)#从目标网络获取各动作估值
            reward+=DISCOUNT_RATE*np.max(Qvalue)#再由当前动作奖励更新估值
        result_est=nn.forward(state)#拿这个目标估值来训练神经网络
        result_est[action]=reward-result_est[action]
        training_set.append([state,result_est])
    epoch_loss=nn.train(training_set)
    loss.append(epoch_loss)#记录当前epoch损失
    
    #输出损失最小的神经网络
    if i==0:
        min_loss=loss[0]
    else:
        if loss[i]<min_loss:
            min_loss=loss[i]
            nn.write_into_csv("min_loss DQN.csv")
    target_nn = copy.deepcopy(nn)#每隔一段时间将训练的网络复制到目标网络作为价值判断标准
    
    if i%10==9:
        net_file='DQN %03d.csv'%(i+1)
        nn.write_into_csv(net_file)
        
            
        anim_file='performance epoch%03d'%(i+1)#动画文件名        
        val_file='trace epoch%03d.png'%(i+1)#轨迹图文件名
        pt=nn_validation(nn,val_env,val_file,anim_file)#生成轨迹图和动画
        pts.append(pt)
        
        #输出损失和分数数据
        with open('loss and pts.txt', 'w') as file:
            file.write("loss:")
            for j in loss:
                file.write("%f,"%j)
            file.write("\n")
            file.write("pts:")
            for j in pts:
                file.write("%f,"%j)
            file.write("\n")


fig, ax = plt.subplots(figsize=(12, 8),dpi=150)
plt.title("Loss Plot",fontsize=24)#图标题,设置字体大小   
plt.plot([i+1 for i in range(len(loss))],loss)


plt.margins(x=0,y=0)
plt.tick_params('both', labelsize=16,length=6)#设置数字大小,length是设置坐标轴刻度线长度    
 
plt.xlabel('epochs', fontsize=18)#坐标轴标题   
plt.ylabel('loss', fontsize=18)
#ax1.set_xticks([20*i for i in range(9)])#设置刻度线打在哪几个数上   
#plt.gca().set_aspect("equal")
fig.savefig("loss.plot.png")
plt.show()

model_result_3D(nn)