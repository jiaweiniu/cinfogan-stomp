# cinfogan initial trjaectory
from scipy import interpolate
from chainer import serializers
import numpy as np
from models import Generator
from chainer import serializers, Variable



def gen_points(gen,n_tests,n_z,obstacles,q_start,q_goal):
    z = np.random.uniform(-2, 2, (n_tests, n_z+n_continuous))
    z = Variable(np.array(z, dtype=np.float32))
    x = np.random.uniform(0.0, 1.0, (n_tests,12));
    x = Variable(x.astype(np.float32))
    xi = gen(z,x)
    return (xi)



if __name__ == '__main__':
    gen = Generator(60,12,6,2,100)
    obstacles = np.random.random((3,2))
    q_start = np.random.random(2)
    q_goal = np.random.random(2)
    n_tests = 500
    n_z = 60
    n_continuous = 2
    serializers.load_npz("../results/models/tmp/21_gen.model",gen)
    print (gen_points(gen,n_tests,n_z,obstacles,q_start,q_goal))

'''
def cinfogan_initial_traj(q_start, q_goal, n_timesteps):
    t = np.linspace(0,1,n_timesteps)       # divided by timesteps times between 0 to 1    
     
    point_1 = np.array([xi_gen[0], xi_gen[1]])
    point_2 = np.array([xi_gen[2], xi_gen[3]])
    point_3 = np.array([xi_gen[4], xi_gen[5]])

    print("Initial trajectory points1")
    print (point_1,point_2,point_3) 
    ξ_x = np.array([q_start[0],point_1[0],point_2[0],point_3[0],q_goal[0]])
    ξ_y = np.array([q_start[1],point_1[1],point_2[1],point_3[1],q_goal[1]])

    xnew = np.linspace(q_start[0], q_start[1], n_timesteps)  
    ynew = interpolate.spline(ξ_x,ξ_y,xnew)

    ξ_0 = np.array([xnew,ynew])
    return ξ_0
'''
