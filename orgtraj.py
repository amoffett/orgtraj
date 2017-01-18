# orgtraj - Organized trajectories
# ---> Simplify storing metadata with featurized trajectories.
#
# Parts directly borrowed from unutbu on http://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe

import numpy as np
import pandas as pd

def prepare(data, features, traj_file, frames = None, states = None, **metadata):
	"""
	Takes a numpy array of features along with trajectory metadata and produces a 
	Pandas dataframe and metadata dictionary.
 	Input:
	- data: numpy array containing a featurized trajectory (N frames X D features)
	- features: list of strings describing the feature dimensions.
	- trajfile: name of the trajectory file features were calculated from.
	- frames: list of frames. Default set to [1, 2, 3, ..., N].
	- states: list of MSM states corresponding to frames. Default set to 0 for all frames.
	- **metadata: keyword arguments defining additional metadata.
	Returns:
	- df: pandas dataframe of features.
	- metadata: dictionary of featurization metadata.
	"""
	if frames == None:
		frames = range(1,data.shape[0]+1)
	if states == None:
		states = [0]*data.shape[0]
	index = pd.MultiIndex(levels=[frames,states],labels=[range(data.shape[0]),range(data.shape[0])],names=['frame','state'])		
	df = pd.DataFrame(data,columns=features,index=index)
	metadata['traj_file'] = traj_file
	return df, metadata

def h5dump(filename, df, dataset = 'traj', **metadata):
	"""
	Writes HDF5 format file of a Pandas DataFrame with a dictionary containing metadata.
	Input:
	- filename: name of file ('*.h5') to write to.
	- df: pandas DataFrame object to store.
	- dataset: name of the dataset to be written.
	- **metadata: dictionary with metadata.
	"""
	store = pd.HDFStore(filename)
	store.put(dataset, df)
	store.get_storer(dataset).attrs.metadata = metadata
	store.close()

def h5load(filename, dataset = 'traj'):
	"""
	Load HDF5 format file created using h5dump.
	Input:
	- filename: name of .h5 file (written using 'h5dump' or the class method 'trajwrite') to load.
	Returns:
	- data: pandas DataFrame object.
	- metadata: dictionary containing the original trajectory filename and additional metadata.
	"""
	with pd.HDFStore(filename) as store: 
		data = store[dataset]
		metadata = store.get_storer(dataset).attrs.metadata
	return data, metadata

def find_frame(point, orgtraj_list):
	"""
	Returns the trajectory name and the frame number of the structure
	closest to the N-dimensional point specified in a feature space. 
	Input:
	- point: list of coordinates of desired point in feature space (by column of pandas dataframes).
	- orgtraj_list: list of orgtraj objects of the desired features.
	Returns: traj_file, frame
	- traj_file: the name of the trajectory with the frame closest to the specific point.    	
	- frame: the mdtraj frame number (starting at 0) for the closest frame to the specific point.
	"""
	datalist = [obj.data for obj in orgtraj_list]
	if len(point) != datalist[0].shape[1]:
		raise ValueError("Point dimension must match feature space dimension.") 
	point = np.array(point)
	def point_dist(dataframe,point):
		array = np.array(dataframe)
		dists = np.sum((array-point)**2,axis=1)
		return dists
	n_traj = np.argmin([np.min(point_dist(i,point)) for i in datalist])
	traj_file = orgtraj_list[n_traj].traj_file
	frame_index = np.argmin(point_dist(datalist[n_traj],point))
	frame = orgtraj_list[n_traj].index[frame_index][0]
	return traj_file, frame

class orgtraj():
	"""
	Class to create organized featurized trajectory objects.
	Attributes:
	- data: pandas DataFrame containing N-frames X D-features array.
		---> first multiindex layer is the frame number.
		---> second multiindex layer is the MSM state number (if set).
		---> columns are labeled by 'features' input.
	- traj_file: file name of trajectory from which the featurized trajectory was created.
	- **metadata: any additional keyword argument will be stored in the orgtraj attribues and
	saved as additional metadata to the .h5 output file if 'trajwrite' is used. 
	"""
	def __init__(self):
		return None
	
	def trajin(self, data, features, traj_file, frames = None, states = None, **metadata):
		"""
        	Takes a numpy array of features along with trajectory metadata and produces a
        	Pandas dataframe and metadata dictionary.
        	Input:
        	- data: numpy array containing a featurized trajectory (N frames X D features)
        	- features: list of strings describing the feature dimensions.
        	- traj_file: name of the trajectory file features were calculated from.
        	- frames: list of frames. Default set to [1, 2, 3, ..., N].
        	- states: list of MSM states corresponding to frames. Default set to 0 for all frames.
        	- **metadata: keyword arguments defining additional metadata.
        	"""	
		data, metadata = prepare(data,features,traj_file,frames=frames,states=states,**metadata)
                self.data = data
		for key, value in metadata.items():
                        setattr(self, key, value)
			
	def trajread(self, infile):
		"""
		Load dataset and metadata from an orgtraj-created .h5 file into orgtraj instance.
		"""
		data, metadata = h5load(infile)
		self.data = data
		for key, value in metadata.items():
                        setattr(self, key, value) 
		
	def trajwrite(self, outfile, dataset = 'traj'):
		"""
		Writes dataset and metadata from orgtraj instance to an .h5 file to be loaded with h5load.
		"""
		meta = dict()
		attrs = vars(self)
		for attr in attrs:
			if attr != 'data':
				meta[attr] = getattr(self,attr)
		h5dump(outfile,self.data,dataset=dataset,**meta)
