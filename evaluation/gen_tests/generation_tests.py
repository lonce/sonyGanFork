import numpy as np
import torch
from numpy.random import randint
from numpy import linspace


class StyleGEvaluationManager(object):
    def __init__(self, model, n_gen=20, get_avg=False):
        self.model = model
        self.att_manager = model.ClassificationCriterion
        self.n_gen = n_gen
        self.get_avg = get_avg
        # self.model.config.ac_gan = True

        self.ref_rand_z = self.model.buildNoiseData(self.n_gen, skipAtts=True)[0]

        # self.ref_rand_z = self.model.buildNoiseData(self.n_gen)[0]
        self.latent_noise_dim = self.model.config.noiseVectorDim
        self.att_dim = self.model.config.categoryVectorDim_G

        self.n_iterp_steps = 10

    def test_random_generation(self):
        gen_batch = self.model.test(self.ref_rand_z,
                               toCPU=True,
                               getAvG=self.get_avg)
        return gen_batch, self.ref_rand_z #LW - added return of batch of latents

    #this function creates a different nonlinear interpolation for each paramter.
    # Each has a different rate at all times, is linear at a differnt point in time, and are maximally spread out at the midpoint
    # spread_interp(dims,t) returns a vectore of len=dims for one point in time
    def spread_interp(self,dims,t) : #spread across all dimensions at a particular time point, t in [0,1] 
        midpoint = int(dims/2)
        spreadvec=np.zeros(dims) #output
        
        powers=np.linspace(5,1,midpoint+1, True)
        for d in range(midpoint) :
            spreadvec[d]= np.power(t, powers[d])
            spreadvec[dims-d-1]= 1-np.power(1-t, powers[d])
        # if dims is odd, do dim {midpoint+1} 
        if dims%2 :
            #print(f"dims is odd, do dim {midpoint+1}")
            spreadvec[midpoint]= np.power(t, powers[midpoint])
        
        return spreadvec

    def test_single_pitch_random_z(self, ppppp="pitch", p_val=55):
        input_z = self.ref_rand_z.clone()
        input_z[:, -self.att_dim:] = torch.zeros(self.att_dim)

        if ppppp in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict[ppppp]
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            att_shift = 0
            for j, att in enumerate(self.att_manager.keyOrder):
                if att in self.att_manager.skipAttDfake: continue
                if att == ppppp: break
                att_shift += self.att_manager.attribSize[j]
            # att_shift = sum(self.att_manager.attribSize[:ppppp_att_indx])
            try:
                ppppp_indx = \
                    self.att_manager.inputDict[ppppp]['values'].index(p_val)
            except ValueError:
                ppppp_indx = randint(ppppp_att_size)
            input_z[:, self.latent_noise_dim + att_shift + ppppp_indx] = 1

        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z

    # chooses a ranodm latent vector, and sweeps across all the conditioned values of the ppppp parameter provided
    def test_single_z_pitch_sweep(self, ppppp="pitch"):
        if ppppp not in self.att_manager.keyOrder: 
            raise AttributeError(f"{ppppp} not in the model's attributes")
        ppppp_att_dict = self.att_manager.inputDict[ppppp]
        ppppp_att_indx = ppppp_att_dict['order']
        ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
        att_shift = 0
        for j, att in enumerate(self.att_manager.keyOrder):
            if att in self.att_manager.skipAttDfake: continue
            if att == ppppp: break
            att_shift += self.att_manager.attribSize[j]
        # att_shift = sum(self.att_manager.attribSize[:ppppp_att_indx])
        input_z = []
        for i in range(ppppp_att_size):
            z = self.ref_rand_z[0].clone()
            z[-self.att_dim:] = torch.zeros(self.att_dim)
            z[-self.att_dim + att_shift + i] = 1
            input_z.append(z)
        input_z = torch.stack(input_z)
        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z

    def test_random_z_pitch_sweep(self, ppppp="pitch"):
        if ppppp not in self.att_manager.keyOrder:
            raise AttributeError(f"{ppppp} not in the model's attributes")
        ppppp_att_dict = self.att_manager.inputDict[ppppp]
        ppppp_att_indx = ppppp_att_dict['order']
        ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
        att_shift = 0
        for j, att in enumerate(self.att_manager.keyOrder):
            if att in self.att_manager.skipAttDfake: continue
            if att == ppppp: break
            att_shift += self.att_manager.attribSize[j]
        # att_shift = sum(self.att_manager.attribSize[:ppppp_att_indx])
        input_z = []
        for i in range(ppppp_att_size):
            z = self.ref_rand_z[i].clone()
            z[-self.att_dim:] = torch.zeros(self.att_dim)
            z[-self.att_dim + att_shift + i] = 1
            input_z.append(z)
        input_z = torch.stack(input_z)
        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z


    #--------------------------------------------------------------------------------------------------
    '''
        Just take four vectors in the latent space (specified with .pt files), uses the first 2 to define one line, 
        the second pair to define another. The two lines define a twisted plane that is sampled as a grid. 
        (If just you want to interpolate along a line (rather than a plane), leave line1zX at default (0) and d1steps at default (1).
        
        The function assumes that network was trained unconditionally (asserting that there is no conditioning segment of 
        the latent vector). 
        
        The dimension parallel to the lines is the "least significant bit" in the nested loop, and samples d0steps points.
        The dimension orthogonal to the lines is the "msb", the outer loop, and samples d1steps points.

        //d1nvar - number of variation at each grip point interp step
        //d1var  - the variance of the gaussian for drawing variations around each grid oint
    '''
    def unconditioned_linear_interpolation(self, line0z0, line0z1, line1z0=0, line1z1=0, d0steps=1, d1steps=1, d1nvar=1, d1var=.03) :
        # initialize a tensor of the right size
        z = self.ref_rand_z[:2, :].clone()
        assert self.att_dim == 0,  "Yo, this network was trained with some conditioning params you MUST include someway"
        
        input_z = []

        # step orthogonal to the lines
        for i in linspace(0., 1., d1steps, True) :
            z[0,:]=(1-i)*line0z0 + i*line1z0
            z[1,:]=(1-i)*line0z1 + i*line1z1
    
            # step parallel to the lines
            for j in linspace(0., 1., d0steps, True):
                p=(1-j)*z[0] + j*z[1]
                input_z.append(p)
                # are we doing variations?
                for nv in range(1, d1nvar):
                    # normal in our latent space
                    p=p+torch.randn(self.latent_noise_dim).cuda()*d1var
                    input_z.append(p)


        input_z = torch.stack(input_z)

        #print(f"Output norms input_z0={torch.norm(input_z[0][:self.latent_noise_dim])} and input_z1={torch.norm(input_z[steps-1][:self.latent_noise_dim])}")

        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z


    #--------------------------------------------------------------------------------------------------

    # interpolates between two provided latent vectors (optionally) at a particular value of the conditioned parameter ppppp
    def test_single_pitch_latent_interpolation(self, ppppp="pitch", p_val=55, z0=None, z1=None, steps=None):
        z = self.ref_rand_z[:2, :].clone()
        if z0 != None :
            z[0,:]=z0
        if z1 != None :
            z[1,:]=z1
        if self.att_dim > 0:
            z[:, -self.att_dim:] = torch.zeros(self.att_dim)

        if steps==None : 
            steps=self.n_iterp_steps

        #print(f"Input Norms z0={torch.norm(z0[:self.latent_noise_dim])} and z1={torch.norm(z1[:self.latent_noise_dim])}")

        if self.att_manager and ppppp in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict[ppppp]
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            try:
                ppppp_indx = \
                    self.att_manager.inputDict[ppppp]['values'].index(p_val)
            except ValueError:
                ppppp_indx = randint(ppppp_att_size)
            
            z[:, self.latent_noise_dim + ppppp_indx] = 1
        
        input_z = []
        for i in linspace(0., 1., steps, True):
        #for i in linspace(-1., 1., steps, True):  #nonlinear interpolation (slow down in the middle)
            #ii=(i*i*i+1)/2                        # [-1,1] -> [0,1]
            input_z.append((1-i)*z[0] + i*z[1])
            # z /= abs(z)
        input_z = torch.stack(input_z)

        #print(f"Output norms input_z0={torch.norm(input_z[0][:self.latent_noise_dim])} and input_z1={torch.norm(input_z[steps-1][:self.latent_noise_dim])}")


        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z




    def test_single_pitch_latent_staggered_interpolation(self, ppppp="pitch", p_val=55, z0=None, z1=None, steps=None, d1nvar=1, d1var=.03):
        z = self.ref_rand_z[:2, :].clone()
        if z0 != None :
            z[0,:]=z0
        if z1 != None :
            z[1,:]=z1
        if self.att_dim > 0:
            z[:, -self.att_dim:] = torch.zeros(self.att_dim)

        if steps==None : 
            steps=self.n_iterp_steps

        #print(f"Input Norms z0={torch.norm(z0[:self.latent_noise_dim])} and z1={torch.norm(z1[:self.latent_noise_dim])}")

        if self.att_manager and ppppp in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict[ppppp]
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            try:
                ppppp_indx = \
                    self.att_manager.inputDict[ppppp]['values'].index(p_val)
            except ValueError:
                ppppp_indx = randint(ppppp_att_size)
            
            z[:, self.latent_noise_dim + ppppp_indx] = 1
        
        input_z = []

        for i in linspace(0., 1., steps, True):  

            for nv in range(d1nvar):
                if nv > 0 :   #the first one will be z unperturbed
                    z[0,:self.latent_noise_dim]=z0[:self.latent_noise_dim]+torch.randn(self.latent_noise_dim).cuda()*d1var
                    z[1,:self.latent_noise_dim]=z1[:self.latent_noise_dim]+torch.randn(self.latent_noise_dim).cuda()*d1var


                spread_i=self.spread_interp(len(z0),i)
                input_z.append(torch.from_numpy(1-spread_i).cuda().float()*z[0] + torch.from_numpy(spread_i).cuda().float()*z[1])
                # z /= abs(z)



        input_z = torch.stack(input_z)

        #print(f"Output norms input_z0={torch.norm(input_z[0][:self.latent_noise_dim])} and input_z1={torch.norm(input_z[steps-1][:self.latent_noise_dim])}")


        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch, input_z



    def test_single_pitch_sph_latent_interpolation(self, ppppp="pitch", p_val=55):

        def get_rand_gaussian_outlier(ndim):
            r = 3
            ph = 2 * np.pi * np.random.rand(ndim)
            cos = [np.cos(p) for p in ph]
            sin = [np.sin(p) for p in ph]
            vector = []
            for i in range(len(ph)):
                if i == 0:
                    vector += [cos[i]]
                elif i == len(ph):
                    vector += [np.prod(sin)]
                else:
                    vector += [np.prod(sin[:i]) * cos[i]]
            input_z = []
            
            # LW: Interpolating the radius??? This isn't spherical interpolation....
            for i in linspace(r, -1*r, self.n_iterp_steps, True):
                input_z.append(torch.from_numpy(np.multiply(i, vector).astype(float)))
            return input_z

        input_z = get_rand_gaussian_outlier(self.latent_noise_dim)
        if self.att_dim > 0:
            input_z = torch.stack(input_z).double()
            input_z = torch.cat([input_z, torch.zeros((input_z.size(0), self.att_dim)).double()], dim=1)

        if ppppp in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict[ppppp]
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            try:
                ppppp_indx = \
                    self.att_manager.inputDict[ppppp]['values'].index(p_val)
            except ValueError:
                ppppp_indx = randint(ppppp_att_size)
            
            input_z[:, self.latent_noise_dim + ppppp_indx] = 1
        # input_z = torch.stack(z)

        gen_batch = self.model.test(input_z.float(), toCPU=True, getAvG=True)
        return gen_batch, input_z.float()


        # I don't think this works the way it should.
        # First, the angle ranges from x to x+2pi
        # Secondly, it doesn't seeme to map out the shortest arc, but looks rather like a multidimensions lissajou function. 
    def test_single_pitch_2point_sph_surface_interpolation(self, pitch=55, z0=None, z1=None, steps=None):

        r = 1
        ph = 2 * np.pi * np.random.rand(self.latent_noise_dim)
        input_z = []
        for i in range(30):
            phi = ph + 2*np.pi*i/30

            cos = [np.cos(p) for p in ph]
            sin = [np.sin(p) for p in ph]
            vector = []
            for i in range(len(ph)):
                if i == 0:
                    vector += [cos[i]]
                elif i == len(ph):
                    vector += [np.prod(sin)]
                else:
                    vector += [np.prod(sin[:i]) * cos[i]]
            zi = np.multiply(r, vector)
            input_z.append(torch.from_numpy(zi.astype(float)))

        if self.att_dim > 0:

            input_z = torch.stack(input_z).double()
            input_z = torch.cat([input_z, torch.zeros((input_z.size(0), self.att_dim)).double()], dim=1)

        if "pitch" in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict['pitch']
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(ppppp_att_size)
            
            input_z[:, self.latent_noise_dim + pitch_indx] = 1
        # input_z = torch.stack(input_z)

        gen_batch = self.model.test(input_z.float(), toCPU=True, getAvG=True)
        return gen_batch, input_z.float()









    #Here is my two-point spherical interpolation. which interpolated radius and angle between arbitrary points
    def qslerp(self, pitch=55, z0=None, z1=None, steps=None):
        if z0 != None :
            z0=z0[:self.latent_noise_dim]
        if z1 != None :
            z1=z1[:self.latent_noise_dim]
        if steps==None : 
            steps=self.n_iterp_steps

        # Normalize to avoid undefined behavior.
        r0=torch.norm(z0)
        r1=torch.norm(z1)

        print(f"Input Norms z0={r0} and z1={r1}")
        
        z0=z0/r0
        z1=z1/r1
        
        d=torch.dot(z0,z1)
        absD=torch.norm(d)

        #theta is the angle between the vectors
        theta = torch.acos(absD)
        sinTheta = torch.sin(theta)

        z=[]
        #for t in np.arange(0, 1., 1./steps):
        for t in linspace(0., 1., steps, True):

        #for i in linspace(-1., 1., steps, True):  #nonlinear interpolation (slow down in the middle)
            #t=(i*i*i+1)/2                        # [-1,1] -> [0,1]


            scale0 = torch.sin((1.0 - t) * theta) / sinTheta;
            scale1 = torch.sin((t * theta)) / sinTheta;

            rscale=(1.0 - t)*r0+t*r1

            #If the dot product is negative, slerp won't take
            #the shorter path. Correct by negating the scale
            if d<0 :
                scale1 = -scale1
                
            z.append(rscale*(scale0 * z0 + scale1 * z1))
            
        z = torch.stack(z)
        print(f"z is on device {z.get_device()}")

        print(f"Output norms z0={torch.norm(z[0])} and z1={torch.norm(z[steps-1])}")

        if self.att_dim > 0:

            #z = torch.stack(z).double()
            z = torch.cat([z, torch.zeros((z.size(0), self.att_dim)).cuda().double()], dim=1)

        if "pitch" in self.att_manager.keyOrder:
            ppppp_att_dict = self.att_manager.inputDict['pitch']
            ppppp_att_indx = ppppp_att_dict['order']
            ppppp_att_size = self.att_manager.attribSize[ppppp_att_indx]
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(ppppp_att_size)
            
            z[:, self.latent_noise_dim + pitch_indx] = 1
        # input_z = torch.stack(input_z)

        gen_batch = self.model.test(z.float(), toCPU=True, getAvG=True)
        return gen_batch, z.float()
