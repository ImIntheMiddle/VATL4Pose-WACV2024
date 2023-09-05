import torch
import torch.utils.data
import numpy as np
from skimage.feature import peak_local_max
from scipy.special import softmax as softmax_fn

class Keypoint_ParallelWrapper(torch.utils.data.Dataset):
    def __init__(self, hm, param, j2i, i2j, links, vl4pose_config):
        self.hm = hm
        self.param = param
        self.j2i = j2i
        self.i2j = i2j
        self.links = links
        self.config = vl4pose_config

    def __len__(self):
        return self.hm.shape[0]

    def __getitem__(self, i):
        joints = {}

        heatmaps = self.hm[i]
        parameters = self.param[i]

        # Initialize keypoints for each node
        for key in self.j2i.keys():

            heatmap = heatmaps[self.j2i[key]]
            loc = peak_local_max(heatmap, min_distance=self.config['min_distance'], num_peaks=self.config['num_peaks'])
            peaks = heatmap[loc[:, 0], loc[:, 1]]
            peaks = softmax_fn(peaks)
            joints[key] = Keypoint(name=key, loc=loc, peaks=peaks)

        # Initialize parent-child relations
        for k, l in enumerate(self.links):

            joints[self.i2j[l[0]]].parameters.append(parameters[k])
            joints[self.i2j[l[0]]].children.append(joints[self.i2j[l[1]]])

        max_ll, trace = joints['head'].run_likelihood()
        return max_ll, trace



class Keypoint(object):
    def __init__(self, name, loc, peaks):
        self.name = name
        self.loc = loc
        self.peaks = peaks
        self.children = []
        self.parameters = []

    def run_likelihood(self):
        """

        :return:
        """
        assert self.name == 'head'

        likelihood_per_location = []
        per_location_trace = []

        for location in range(self.loc.shape[0]):
            log_ll = np.log(self.peaks[location])

            per_child_trace = []
            for child in range(len(self.children)):
                child_ll, joint_trace = self.children[child].compute_likelihood_given_parent(self.loc[location], self.parameters[child])
                log_ll += child_ll
                per_child_trace.append(joint_trace)

            likelihood_per_location.append(log_ll)
            per_location_trace.append(per_child_trace)

        likelihood_per_location = np.array(likelihood_per_location)

        return_trace = {}
        for child_trace in per_location_trace[np.argmax(likelihood_per_location)]:
            return_trace.update(child_trace)

        return_trace[self.name] = np.argmax(likelihood_per_location)
        return_trace['{}_uv'.format(self.name)] = self.loc[np.argmax(likelihood_per_location)]
        return np.sum(likelihood_per_location), return_trace


    def compute_likelihood_given_parent(self, parent_location, gaussian_params):
        """

        :param parent_location:
        :param gaussian_params:
        :return:
        """

        likelihood_per_location = []
        per_location_trace = []

        for location in range(self.loc.shape[0]):
            log_ll = np.log(2 * np.pi) + gaussian_params[1]
            log_ll += (gaussian_params[0] - np.linalg.norm(parent_location - self.loc[location]))**2 * np.exp(-gaussian_params[1])
            log_ll *= -0.5
            log_ll += np.log(self.peaks[location])

            if len(self.children) == 0:
                likelihood_per_location.append(log_ll)

            else:
                per_child_trace = []
                for child in range(len(self.children)):
                    child_ll, joint_trace = self.children[child].compute_likelihood_given_parent(self.loc[location], self.parameters[child])
                    log_ll += child_ll
                    per_child_trace.append(joint_trace)

                likelihood_per_location.append(log_ll)
                per_location_trace.append(per_child_trace)

        likelihood_per_location = np.array(likelihood_per_location)

        if len(self.children) == 0:
            return np.sum(likelihood_per_location), {self.name: np.argmax(likelihood_per_location),
                                                         '{}_uv'.format(self.name): self.loc[np.argmax(likelihood_per_location)]}

        return_trace = {}
        for child_trace in per_location_trace[np.argmax(likelihood_per_location)]:
            return_trace.update(child_trace)

        return_trace[self.name] = np.argmax(likelihood_per_location)
        return_trace['{}_uv'.format(self.name)] = self.loc[np.argmax(likelihood_per_location)]
        return np.sum(likelihood_per_location), return_trace