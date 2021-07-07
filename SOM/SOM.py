import torch
import numpy as np
from evaluation.gen_tests.generation_tests import StyleGEvaluationManager


'''
    Class for the SOM
    2D grid of weights that iteratively move toward regions of change of some function on spatical coords.
    Create the initial grid with the number of rows (ylength) and columns (xlength) you want, together
        with 4 points in a space of arbitrary dimensions.
        
    If vidims is an integer, then it will ignore anything > that number of latent dims for computing differences for visualization (e.g. the often one-hot conditioning parmaters which may not be used for SOM remapping anyway)

    To adjust the map,
    Iterate: getDifferences(values of neighbors) -> diffVect, -> update locations
''' 
class kmap:
    def __init__(self,   ylength,  xlength,  line0z0=np.array([0,0]), line0z1=np.array([0,1]), line1z0=np.array([1,0]), line1z1=np.array([1,1]), minPitch=None, maxPitch=None, pitchDim=None, vizdims="all"):
        
        self.x, self.y = np.meshgrid(np.linspace(0,xlength-1, xlength, True), np.linspace(0, ylength-1, ylength, True))

        z = np.zeros((2, len(line0z0)))  #endpoints for each row      
        grid_z=[] # first append to list, then later stack to np.array

        # step orthogonal to the lines
        for i in np.linspace(0, 1, ylength, True):
            z[0,:]=(1-i)*line0z0 + i*line1z0
            z[1,:]=(1-i)*line0z1 + i*line1z1

            row_z = [] #  the pythonic way: first append to list, then later stack to np.array

            # step parallel to the lines
            for j in np.linspace(0, 1,  xlength, True) :
                p=(1-j)*z[0] + j*z[1]
                row_z.append(p)
            row_z=np.stack(row_z, axis=0)

            grid_z.append(row_z)
            
        #self.weights is the matrix of SOM "weights"
        self.weights = np.stack(grid_z, axis=0)
        print(f"weights.shape is {self.weights.shape}")
        
        # for visualization, weights on the 2D submanifold of the mesh
        if vizdims=="all" :
            vizdims=self.weights.shape[2]
        xspacing = np.linalg.norm(self.weights[0,1,:vizdims]-self.weights[0,0,:vizdims])
        yspacing = np.linalg.norm(self.weights[1,0:vizdims]-self.weights[0,0:vizdims])
        print(f"xspacing is {xspacing}. and yspacing is {yspacing}")
        self.weights2D=np.transpose(np.indices((ylength,xlength)),axes=(1,2,0))*np.array([xspacing, yspacing])
        print(f"weights2D.shape is {self.weights2D.shape}")
        
    
    def location(self, a, b) :
        return self.weights[a,b,:]
    
    #used for plotting
    #like np grid mesh, returns a 2D array of x values, and a 2D array of y values
    def locationMesh(self) :
        return self.weights[:,:,0], self.weights[:,:,1]
    
        
    '''
        returns a 2x2 matrix of vectors of length 8, with the differences between the value of a function at their respective weights 
        You can limit the non-zero mesh direction diffrerences with the dirs list (0 coresponds to 2pi, the reset index increments of Pi/4)
    '''         
    def getDifferences(self, vals, directions=[0,1,2,3,4,5,6,7], metric="L2") :
        #first duplicate values along the edges so we can roll. Implies diffs along edges will be 0.
        #assert vals.shape[:2] == self.weights.shape[:2], f"vals.shape={vals.shape}, but mesh shape is {self.weights.shape}"
        dirs=8 # counter clockwise in [0,360,step=45] . Redunant since links are symmetric        
        rows, cols = self.weights.shape[:2]
        rows=rows+2
        cols=cols+2
        vals=np.pad(vals, 1, mode='edge') # duplicate edge
        diff=np.zeros((rows,cols, dirs))
        
        rolldir=[(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
    
        for d in directions : #range(dirs) :
            if len(vals.shape) > 2 :
                sumdims= tuple(range(2, len(vals.shape))) # if vals is not a scalar, sum over all the dimensions
                if metric == "L1" :
                     diff[:,:,d] = np.sum(np.abs(vals-np.roll(vals, rolldir[d], axis=(0,1))), axis=sumdims)
                else :
                    diff[:,:,d] = np.sqrt(np.sum(np.square(np.abs(vals-np.roll(vals, rolldir[d], axis=(0,1)))), axis=sumdims))
            else :
                diff[:,:,d] = np.abs(vals-np.roll(vals, rolldir[d], axis=(0,1)))
                                    
        return diff[1:rows-1,1:cols-1] # strip padding

        
    '''
    Turns vector of differences in 8 directions into a weightdims direction vector, the vector sum of the
        differences in each of the neighbor's directions
    '''
    def diffVect(self, wmatrix, diffs, clampedges=False, directions=[0,1,2,3,4,5,6,7] ) :
        
        #first duplicate values along the edges so we can roll. Means diffs along edges to outside neighbors will be 0.
        #assert vals.shape[:2] == self.weights.shape[:2], f"vals.shape={vals.shape}, but mesh shape is {self.weights.shape}"
        #dirs=8 # counter clockwise in [0,360,step=45] . Redunant since links are symmetric        
        rows, cols, weightdims = wmatrix.shape
        prows=rows+2
        pcols=cols+2
        
        #Pad weights (rows,cols,coordarray) with edge coordarrays
        pos= np.pad(wmatrix, ((1,1),(1,1),(0,0)), mode='edge') # duplicate edge so that difference in pos = 0 (tho doesn't matter if diffs in that direction is 0 anyway)
        #Pad diffs (rows, cols, [mags in each of 8 dims]) by duplicating the mag array along the edges
        diffs=np.pad(diffs,((1,1),(1,1),(0,0)), mode='edge')
        
        #roll *from* 0 degrees to 315 degrees incrementing by 45 degrees
        rolldir=[(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
        
        # ----  first in the real high-D space -------------------------
        dv=np.zeros((prows,pcols, weightdims))
        
        
        
        # multiply difference between neighbors by the distance along the edge to the neighbor
        for d in directions : #range(dirs) :
            #roll one mesh unit in possibly both directions to get each of the eight neighbors
            neighborv=(np.roll(pos, rolldir[d], axis=(0,1))-pos)
            dv = dv + diffs[:,:,d,np.newaxis]*neighborv

        
        #print(f"New we have DV - the sum of all the mags in the eight different different directions. It's shape is {dv.shape}") 
        
        #OK, shave off the padding on dv
        dv=dv[1:rows+1,1:cols+1]

        if clampedges :
            # next, for the four edges of the 2D manifold, let's project dv onto the edge so they don't move of their lines.
            d0=wmatrix[0,cols-1]-wmatrix[0,0]
            d0unit=d0/np.linalg.norm(d0)
            dv[0,:] = np.multiply(np.dot(dv[0,:], d0)[:,np.newaxis], np.tile(d0unit, (cols,1)))

            d4=wmatrix[rows-1,cols-1]-wmatrix[rows-1,0]
            d4unit=d4/np.linalg.norm(d4)
            dv[rows-1,:] = np.multiply(np.dot(dv[rows-1,:], d4)[:,np.newaxis], np.tile(d4unit, (cols,1)))

            d2=wmatrix[rows-1,0]-wmatrix[0,0]
            d2unit=d2/np.linalg.norm(d2) 
            dv[:,0] = np.multiply(np.dot(dv[:,0], d2)[:,np.newaxis], np.tile(d2unit, (rows,1)))

            d6=wmatrix[rows-1,cols-1]-wmatrix[0,cols-1]
            d6unit=d6/np.linalg.norm(d6)
            dv[:,cols-1] = np.multiply(np.dot(dv[:,cols-1], d6)[:,np.newaxis], np.tile(d6unit, (rows,1)))
                   
                           
        #pin corners (not forgetting to remove padding first)
        dv[0,0]=dv[rows-1,0]= dv[0,cols-1]=dv[rows-1,cols-1]=np.zeros(weightdims)
           
        return dv

                  

    def updateLocations(self, wmatrix, dv, step) :
        '''
            Move the weights along the direction vector by some increpemental step size
        '''
        rows, cols, weightdims = wmatrix.shape
        
        #movable= np.full((rows,cols, weightdims), True, dtype=bool)
        #movable[0,0]=movable[rows-1,0]= movable[0,cols-1]=movable[rows-1,cols-1]= numpy.full((weightdims), False)
       # self.weights = np.where(movable, self.weights+step*dv, self.weights)        
        wmatrix = wmatrix+step*dv
        return wmatrix
    
    # vals is the value of a function of the mesh location that will drive the mesh location (weight) changes
    def weightUpdate(self, vals, step=.025, clampedges=True, directions=[0,1,2,3,4,5,6,7]) :
        '''
            Update the mesh locations (weights) based on the differences between neighboring
            values of the function at mesh locations.
            Return: Sum of changes made to the locations of the mesh points 
        '''
        # get the differences beteen the values at a node and it neighbors
        # diffm=p1.getDifferences(G, directions=[0,4])
        diffm=self.getDifferences(vals, directions)

        # get the direction we want to move the mesh points
        dv=self.diffVect(self.weights, diffm, clampedges=clampedges, directions=directions)
        dv2D=self.diffVect(self.weights2D, diffm, clampedges=clampedges, directions=directions)

        changesum=np.sum(np.linalg.norm(dv2D, axis=2, keepdims=True))

        # update the mesh points
        #CLAMPED EDGES REQUIRE SMALLER STEP SIZE TO AVOID NANs
        self.weights=self.updateLocations(self.weights, dv, step)
        self.weights2D=self.updateLocations(self.weights2D, dv2D, step)
        
        return changesum, diffm
    

############################################################################################

'''
    Change the pitch value of a latent vector that is already concatenated with the dims for conditioned params

    (if the .pt files you are loading were generated be a network trained with pitch conditioning, then they will have 
    a one-hot vector augmenting the latent vector. Here we set the one-hot segment to the value representing the p_val)
'''
def setPitch(z, evman, cparam="pitch", p_val=58) :
    if evman.att_dim > 0:
        z[ -evman.att_dim:] = torch.zeros(evman.att_dim)
    if evman.att_manager and cparam in evman.att_manager.keyOrder:
        p_att_dict = evman.att_manager.inputDict[cparam]
        p_att_indx = p_att_dict['order']
        p_att_size = evman.att_manager.attribSize[p_att_indx]
        try:
            p_indx = evman.att_manager.inputDict[cparam]['values'].index(p_val)
        except ValueError:
            p_indx = randint(p_att_size)

        z[ evman.latent_noise_dim + p_indx] = 1
        return z

