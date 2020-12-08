import numpy as np 
import os
from nilearn.image import index_img, smooth_img
from torchvision import transforms, utils
import pickle 


def storeData(object, file_name, root_dir):
    with open(root_dir+file_name, 'wb') as f:
        pickle.dump(object, f)					 
        f.close() 

def loadData(file_name, root_dir): 
    with open(root_dir+file_name, 'rb') as f:
        db = pickle.load(f) 
        f.close()
        return db



# In case we wanted to do experiments with ILC inter subjects.
# This class is also good when we want to deal with different runs from the same
# subject.
class fmriDatasetAllSubjects(Dataset):
    """fMRI dataset when all subjects data is mixed."""

    def __init__(self, root_dir, files_paths, list_of_partitions, list_of_labels, format, transform=None):
        # event_file could be added when real data is available.
        # add this argument later: tsv_file
        """
        Args:
            files_paths (string): Path to the (fmri) file (nifti or npy).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample, i.e. flattening and subsampling.
        """
        self.root_dir = root_dir

        if format is 'nifti':
            self.number_of_subjects = len(files_paths)
            self.labels = np.concatenate([list_of_labels[i] for i in range(self.number_of_subjects)], axis=0)

            # list of NiftiImage objects for different subjects, i.e., if there are "two" 
            # subjects, then X contains "two" 4-D images of shape (x_dim,y_dim,z_dim,time)
            X = []

            # list of events
            Y = []

            # This loop goes over the nifti images of the subjects
            for index, image_path in enumerate(files_paths):
                # load image and remove nan and inf values.
                # applying smooth_img to an image with fwhm=None simply cleans up
                # non-finite values but otherwise doesn't modify the image.
                self.path = os.path.join(self.root_dir, image_path)
                image = smooth_img(self.path, fwhm=None)
                X.append(image)

            

            
            self.subject_frames = np.concatenate([X[i].get_fdata()[:,:,:,list_of_partitions[i]] for i in range(self.number_of_subjects)], axis=3)

        else:
            self.labels = list_of_labels

            self.path = os.path.join(self.root_dir, files_paths)
        
            images = np.load(self.path, encoding='bytes')
            images = images[list_of_partitions,:,:,:]
            images = np.reshape(images, (images.shape[1],images.shape[2],images.shape[3],images.shape[0]))

            
            self.subject_frames = images

        self.transform = transform

    def __len__(self):
        return (self.subject_frames).shape[3]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.subject_frames[:,:,:,idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]

# In case we wanted to do experiments with ILC intra subjects
class fmriDatasetSubject(Dataset):
    """fMRI dataset for each subject."""

    def __init__(self, root_dir, file_path, list_IDs, labels, format, transform=None):
        # event_file could be added when real data is available.
        # add this argument later: tsv_file
        """
        Args:
            file_path (string): Path to the (fmri) file (nifti or npy).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample, i.e. flattening and subsampling.
        """
        
        self.root_dir = root_dir
        self.file_path = file_path
        self.labels = labels
        self.path = os.path.join(self.root_dir, self.file_path)
        
        if format is 'nifti':
            self.subject_frames = smooth_img(self.path, fwhm=None).get_fdata()
            self.subject_frames = self.subject_frames[:,:,:,list_IDs]

        else:

            images = np.load(self.path, encoding='bytes')
            images = images[list_IDs,:,:,:]
            images = np.reshape(images, (images.shape[1],images.shape[2],images.shape[3],images.shape[0]))
            self.subject_frames = images


        self.transform = transform

    def __len__(self):
        return (self.subject_frames).shape[3]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.subject_frames[:,:,:,idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]

class Flatten(object):
    """flatten the 3D image of a timestep.

    Args:
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        dim_x = sample.shape[0]
        dim_y = sample.shape[1]
        dim_z = sample.shape[2]

        sample = sample.reshape(dim_x*dim_y,dim_z)
        sample = sample.reshape(sample.shape[0]*dim_z)
        return sample

class Subsample(object):
    """subsample the 3D image of a timestep.

    Args:
    """

    def __init__(self, subsample_rate_x,subsample_rate_y,subsample_rate_z):
        self.rate_x = subsample_rate_x
        self.rate_y = subsample_rate_y
        self.rate_z = subsample_rate_z

    def __call__(self, sample):

        dim_x = sample.shape[0]
        dim_y = sample.shape[1]
        dim_z = sample.shape[2]

        indexes_x = range(0,dim_x,self.rate_x)
        indexes_y = range(0,dim_y,self.rate_y)
        indexes_z = range(0,dim_z,self.rate_z)

        sample = sample[indexes_x,:,:]
        sample = sample[:,indexes_y,:]
        sample = sample[:,:,indexes_z]

        return sample
