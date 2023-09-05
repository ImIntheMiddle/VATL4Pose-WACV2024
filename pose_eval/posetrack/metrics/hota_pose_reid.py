
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from ..utils import count_valid_joints
import math
from ..utils import TrackEvalException
from tqdm import tqdm

EPS = 1 / 1000

class HOTAReidKeypoints(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, n_joints=15):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_fields = ['HOTA_TP(0)', 'HOTA_FN(0)', 'HOTA_FP(0)']
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'FragA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields + self.integer_fields
        self.summary_fields = self.float_array_fields + self.float_fields
        self.joint_names = ['Nose', 'Neck', 'Head', 'LS', 'RS', 'LE', 'RE', 'LW', 'RW', 'LH', 'RH', 'LK', 'RK', 'LA',
                            'RA']
        self.n_joints = n_joints

    def distance2sim(self, distance_mtx: np.ndarray):
        # as we have head normalized l2 distance between gt and det
        # we have a match if dist <= 0.5
        # for that reason, we set any distance > 0.5 to a similarity value of 0
        m = (-1 / 0.5001)
        sim = np.maximum(m * distance_mtx + 1, 0)

        return sim

    @_timing.time
    def eval_sequences(self, processed_seqs, global_gt_ids, global_pr_ids, total_frames):
        MAX_FRAMES_PER_SEQUENCE = 300

        ######################
        # Global Definitions #
        ######################
        
        num_sequences = len(processed_seqs)
        # Initialise gloab results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels), self.n_joints), dtype=float)
        for field in self.float_fields:
            res[field] = 0
        for field in self.integer_fields:
            res[field] = np.zeros([self.n_joints], dtype=int)

        # First, we initialize global vairiables
        num_gt_ids = len(global_gt_ids)
        num_pr_ids = len(global_pr_ids) 

        # Variables counting global association
        #potential_matches_count = np.zeros((num_gt_ids, num_pr_ids, self.n_joints)) 
        potential_matches_count = np.zeros((len(self.array_labels), num_gt_ids, num_pr_ids, self.n_joints))
        gt_id_count = np.zeros((num_gt_ids, 1, self.n_joints))
        tracker_id_count = np.zeros((1, num_pr_ids, self.n_joints))
        
        #fragmentation_count = np.zeros((len(self.array_labels), num_gt_ids, num_pr_ids, self.n_joints), dtype=int)
        #gt_frames = np.ones((len(self.array_labels), num_gt_ids, self.n_joints), dtype=int) * -1 
        #fragmentation_frames = np.ones_like(fragmentation_count) * -1 

        last_matched_id = np.ones((len(self.array_labels), num_gt_ids, self.n_joints), dtype=int) * -1 
        num_gt_fragmentations = np.zeros((len(self.array_labels), num_gt_ids, self.n_joints), dtype=int)
        tp_fragmentation_count = np.zeros((len(self.array_labels), num_gt_ids, num_pr_ids, self.n_joints), dtype=int)
        fragments = np.empty(shape=(len(self.array_labels), num_gt_ids, num_pr_ids, self.n_joints), dtype=object)

        ##########################################################
        # Evaluate each sequence and calculate potential matches #
        ##########################################################
        
        # Iterate over sequences and measure associations 
        for seq_idx, (seq_name, seq_data) in enumerate(processed_seqs.items()): 
            if seq_data['num_tracker_dets'] == 0:
                res['HOTA_FN'] += seq_data['num_gt_joints'][None, :] * np.ones((len(self.array_labels), self.n_joints), dtype=float)
                res['LocA'] += np.ones((len(self.array_labels), self.n_joints), dtype=float)
                res['LocA(0)'] += np.ones((self.n_joints), dtype=float)
                res = self._compute_final_fields(res, compute_avg=True)
                continue # go to next sequence 
            
            if seq_data['num_gt_dets'] == 0:
                res['HOTA_FP'] += seq_data['num_tracker_joints'][None, :] * np.ones((len(self.array_labels), self.n_joints), dtype=float)
                res['LocA'] += np.ones((len(self.array_labels), self.n_joints), dtype=float)
                res['LocA(0)'] += np.ones((self.n_joints), dtype=float)
                res = self._compute_final_fields(res, compute_avg=True)
                continue # go to next sequence 
            
            seq_data['keypoint_similarity'] = list()
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(seq_data['gt_ids'], seq_data['tracker_ids'])):
                keypoint_distances = seq_data['keypoint_distances'][t]
                keypoint_sim = self.distance2sim(keypoint_distances) 
                seq_data['keypoint_similarity'].append(keypoint_sim)

                gt_joints = seq_data['gt_dets'][t]
                tracker_joints = seq_data['tracker_dets'][t]

                valid_gt_kpts = (gt_joints[:, :, 0] > 0) & (gt_joints[:, :, 1] > 0)
                valid_tracker_kpts = (tracker_joints[:, :, 0] > 0) & (tracker_joints[:, :, 1] > 0)

                gt_id_count[gt_ids_t, 0] += valid_gt_kpts
                
                # calculate the total number of dets per joint for each gt_id and tracker_id
                if len(tracker_ids_t) > 0:
                    tracker_id_count[0, tracker_ids_t] += valid_tracker_kpts

                    for a, alpha in enumerate(self.array_labels):
                        for j in range(self.n_joints):
                            for row_ind, col_ind in np.argwhere(keypoint_sim[:, :, j] >= alpha):
                                potential_matches_count[a, gt_ids_t[row_ind], tracker_ids_t[col_ind], j] += 1
            
        # global alignemtn score ^= A_max
        global_alignment_score = potential_matches_count / (np.maximum(1, gt_id_count + tracker_id_count - potential_matches_count))
        matches_counts = np.zeros_like(potential_matches_count)

        ###################
        # Unique Matching #
        ###################
        for seq_idx, (seq_name, seq_data) in enumerate(tqdm(processed_seqs.items())):
            # Calculate scores for each timestep
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(seq_data['gt_ids'], seq_data['tracker_ids'])):

                num_gt_joints = seq_data['num_gt_joints']
                num_tracker_joints = seq_data['num_tracker_joints']

                # calculate num dets per joint class for frame t
                gt_kpts_t = seq_data['gt_dets'][t]
                det_kpts_t = seq_data['tracker_dets'][t]
                num_gt_joints_t = np.sum((gt_kpts_t[:, :, 0] > 0) & (gt_kpts_t[:, :, 1] > 0), axis=0)
                num_det_joints_t = np.sum((det_kpts_t[:, :, 0] > 0) & (det_kpts_t[:, :, 1] > 0), axis=0)

                # Deal with the case that there are no gt_det/tracker_det in a timestep.
                if len(gt_ids_t) == 0:
                    for a, alpha in enumerate(self.array_labels):
                        res['HOTA_FP'][a] += num_tracker_joints
                    continue
                if len(tracker_ids_t) == 0:
                    for a, alpha in enumerate(self.array_labels):
                        res['HOTA_FN'][a] += num_gt_joints
                    continue

                # Get matching scores between pairs of dets for optimizing HOTA
                similarity = seq_data['keypoint_similarity'][t]

                # Hungarian algorithm to find best matches
                for j in range(self.n_joints):
                    # Calculate and accumulate basic statistics
                    for a, alpha in enumerate(self.array_labels):
                        # Do the matching! We priortize TP over accuracy 
                        ms = ((similarity[:, :, j] >= alpha) / EPS).astype(float) 
                        ms += similarity[:, :, j]
                        
                        match_rows, match_cols = linear_sum_assignment(ms, maximize=True)

                        actually_matched_mask = similarity[match_rows, match_cols, j] >= alpha - np.finfo('float').eps
                        alpha_match_rows = match_rows[actually_matched_mask]
                        alpha_match_cols = match_cols[actually_matched_mask]
                        num_matches = len(alpha_match_rows)
                        res['HOTA_TP'][a][j] += num_matches

                        res['HOTA_FN'][a][j] += num_gt_joints_t[j] - num_matches
                        res['HOTA_FP'][a][j] += num_det_joints_t[j] - num_matches
                        if num_matches > 0:
                            res['LocA'][a][j] += sum(similarity[alpha_match_rows, alpha_match_cols, j])
                            matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols], j] += 1

                        # =============================== Count Fragmentations ==============================
                        matched_gt_ids = gt_ids_t[alpha_match_rows]
                        matched_det_ids = tracker_ids_t[alpha_match_cols]

                        # get gt id and pr id with a new fragmentation
                        last_matched_id_a = last_matched_id[a, matched_gt_ids, j]
                        fragmentation_idxs = last_matched_id_a != matched_det_ids
                        gt_id_w_fragmentation = matched_gt_ids[fragmentation_idxs] 
                        pr_id_w_fragmentation = matched_det_ids[fragmentation_idxs]

                        # update fragmentation counts 
                        last_matched_id[a, gt_id_w_fragmentation, j] = matched_det_ids[fragmentation_idxs]
                        num_gt_fragmentations[a, gt_id_w_fragmentation, j] += 1
                        tp_fragmentation_count[a, gt_id_w_fragmentation, pr_id_w_fragmentation, j] += 1

                        # count tp inside current fragment! 
                        fragment_indices = np.maximum(0, tp_fragmentation_count[a, matched_gt_ids, matched_det_ids, j] - 1)

                        if len(matched_gt_ids) > 0 and len(matched_det_ids) > 0:

                            current_fragments = fragments[a, matched_gt_ids, matched_det_ids, j]
                            for curr_frag_idx, fragment_idx in zip(range(len(current_fragments)),  fragment_indices):

                                # fill uninitialized fragment arrays
                                if not isinstance(current_fragments[curr_frag_idx], np.ndarray):
                                    current_fragments[curr_frag_idx] = np.zeros([1])

                                # add new fragment, if required
                                size_diff = len(current_fragments[curr_frag_idx]) - (fragment_idx + 1)
                                if size_diff < 0:
                                    current_fragments[curr_frag_idx] = np.pad(current_fragments[curr_frag_idx],  (0, abs(size_diff)), 'constant',  constant_values=0)

                                # update fragment count
                                current_fragments[curr_frag_idx][fragment_idx] += 1 

                            # update fragments 
                            fragments[a, matched_gt_ids, matched_det_ids, j] = current_fragments
                            #fragments[a, matched_gt_ids, matched_det_ids, j, fragment_indices] += 1

        # Calculate global! association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(tqdm(self.array_labels)):
            matches_count = matches_counts[a]
            '''
            ass_a in the code is A(c) in the equation of the paper.
            The second 'matches_count' is needed because of the 'sum over TPs'
            the ass_a matrix in the code records the A score for each combination of GT and predicted tracks.
            However, the sum you see there is not over tracks but of TPs. The 'matches count' matrix tells us 
            how many TPs there are for each GT/pred track combination. Thus to sum over all the TPs,
            we mltiply by this matrix
            '''
            tpa_fna_fpa = np.maximum(1, gt_id_count + tracker_id_count - matches_count)

            ass_a = matches_count / tpa_fna_fpa
            res['AssA'][a] = np.sum(np.sum(matches_count * ass_a, 0), 0) / np.maximum(1, res['HOTA_TP'][a])

            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(np.sum(matches_count * ass_re, 0), 0) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(np.sum(matches_count * ass_pr, 0), 0) / np.maximum(1, res['HOTA_TP'][a])

            curr_fragments = fragments[a]
            frag = np.zeros(curr_fragments.shape)
            for gt_idx in range(curr_fragments.shape[0]):
                for dt_idx in range(curr_fragments.shape[1]):
                    for j in range(curr_fragments.shape[2]):
                        if isinstance(curr_fragments[gt_idx,  dt_idx, j], np.ndarray):
                            frag[gt_idx,  dt_idx,  j] = (curr_fragments[gt_idx, dt_idx, j] ** 2 / tpa_fna_fpa[gt_idx,  dt_idx,  j]).sum()

            #frag = curr_fragments / tpa_fna_fpa[:, :, :]
            #frag = np.sum(curr_fragments * frag, -1)
            res['FragA'][a] = np.sum(np.sum(frag, 0), 0) / np.maximum(1, res['HOTA_TP'][a])

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res, compute_avg=True)
        return res

    @_timing.time
    def eval_sequence(self, data):
        raise NotImplemented("Not implemented")

    def _compute_final_fields(self, res, compute_avg):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['RHOTA'] = np.sqrt(res['DetRe'] * res['AssA'])
        
        res['FA-HOTA'] = np.sqrt(res['DetA'] * np.sqrt(res['AssA'] * res['FragA']))
        res['FA-RHOTA'] = np.sqrt(res['DetRe'] * np.sqrt(res['AssA'] * res['FragA']))

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)'] * res['LocA(0)']
        res['HOTA_TP(0)'] = res['HOTA_TP'][0]
        res['HOTA_FN(0)'] = res['HOTA_FN'][0]
        res['HOTA_FP(0)'] = res['HOTA_FP'][0]

        # calculate overall average!
        if compute_avg:
            for k, v in res.items():
                if k in self.float_array_fields:
                    if isinstance(v, np.ndarray):
                        avg = np.mean(v, axis=1, keepdims=True)
                        res[k] = np.append(v, avg, axis=1)
                if k in self.integer_array_fields:
                    res[k] = np.append(v, np.sum(v, axis=1, keepdims=True), axis=1)
                if k in self.float_fields:
                    res[k] = np.append(v, np.mean(v, keepdims=True), axis=0)
                if k in self.integer_fields:
                    res[k] = np.append(v, np.sum(v, axis=0, keepdims=True), axis=0)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA', 'FragA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res, compute_avg=False)
        return res

    def print_paper_summary(self, table_res, tracker, cls):
        print('')
        print("Latex Paper Summary")
        order = ['DetPr', 'DetRe', 'DetA', 'AssPr', 'AssRe', 'AssA', 'LocA', 'FragA', 'HOTA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        order = order + ['LocA(0)', 'HOTALocA(0)', 'HOTA(0)', 'HOTA_TP', 'HOTA_FP', 'HOTA_FN']
        order = order + ['HOTA_TP(0)', 'HOTA_FP(0)', 'HOTA_FN(0)']

        order_print = [metric.replace('_', '\\_') for metric in order]

        self._row_print_latex(['Summary'] + order_print)
        output = ['']
        for metric in order:
            metric_results = table_res['COMBINED_SEQ'][metric]
            summary_res = self._summary_result(metric, metric_results)
            output.append(summary_res[-1])

        self._row_print_latex(output)

    def print_table(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + '->evaluating: ' + tracker + ':'])
        seq_names = list(table_res.keys())

        # ToDO: potential removal candidate
        metric_names = list(table_res[seq_names[0]].keys())
        self._row_print(['Sequence'] + metric_names, space=15)
        for seq, results in sorted(table_res.items()):
            if seq == 'COMBINED_SEQ':
                continue
            seq_results = []
            for metric, metric_results in results.items():
                summary_res = self._summary_result(metric, metric_results)
                seq_results.append(summary_res[-1])
            self._row_print([seq] + seq_results, space=15)

        self._row_print([metric_name + '->Summary: ' + tracker + ':'])
        self._row_print(['\t'] + self.joint_names + ['Total'])
        self._row_print(['\tMetric'])
        for metric, metric_results in table_res['COMBINED_SEQ'].items():
            summary_res = self._summary_result(metric, metric_results)
            self._row_print([f"\t{metric}"] + summary_res)

    def print_table_detailed(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + '->evaluating: ' + tracker + ':'])
        for seq, results in sorted(table_res.items()):
            if seq == 'COMBINED_SEQ':
                continue
            # print header
            self._row_print([f'{seq}'])
            self._row_print(['\tMetric'] + self.joint_names + ['Total'])
            for metric, metric_results in results.items():
                summary_res = self._summary_result(metric, metric_results)
                self._row_print([f"\t{metric}"] + summary_res)

        self._row_print([metric_name + '->Summary: ' + tracker + ':'])
        self._row_print(['\t'] + self.joint_names + ['Total'])
        self._row_print(['\tMetric'])
        for metric, metric_results in table_res['COMBINED_SEQ'].items():
            summary_res = self._summary_result(metric, metric_results)
            self._row_print([f"\t{metric}"] + summary_res)

    def _summary_result(self, metric, result):
        vals = []
        # we have a float value per joint
        if metric in self.float_fields:
            for j in range(len(result)):
                vals.append("{0:1.5g}".format(100 * result[j]))
        # we have an array for each joint
        elif metric in self.float_array_fields:
            for j in range(result.shape[1]):
                vals.append("{0:1.5g}".format(100 * np.mean(result[:, j])))
        # we have an array for each joint
        elif metric in self.integer_array_fields:
            for j in range(result.shape[1]):
                vals.append("{0:d}".format(np.mean(result[:, j]).astype(int)))
        elif metric in self.integer_fields:
            for j in range(result.shape[0]):
                vals.append("{0:d}".format(result[j].astype(int)))
        else:
            raise TrackEvalException(f"Unknown metric {metric}")

        return vals

    @staticmethod
    def _row_print(*argv, **kwargs):
        space = kwargs['space'] if 'space' in kwargs else 10

        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = '%-25s' % argv[0]
        for v in argv[1:]:
            to_print += f'%-{space}s' % str(v)
        print(to_print)

    @staticmethod
    def _row_print_latex(*argv):
        to_print = HOTAReidKeypoints._row_to_latex(*argv)
        print(to_print)

    @staticmethod
    def _row_to_latex(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = f'{argv[0]} &'
        for v in argv[1:-1]:
            to_print += f' {v} &'
        to_print += f' {argv[-1]}'

        return to_print

    def summary_results(self, table_res):
        """Returns a simple summary of final results for a tracker"""
        ret = dict()
        for metric, result in table_res['COMBINED_SEQ'].items():
            ret[metric] = self._summary_result(metric, result)

        return ret

    def paper_summary_results(self, table_res):
        order = ['DetPr', 'DetRe', 'DetA', 'AssPr', 'AssRe', 'AssA', 'LocA', 'FragA', 'HOTA', 'RHOTA', 'FA-HOTA', 'FA-RHOTA']
        order = order + ['LocA(0)', 'HOTALocA(0)', 'HOTA(0)', 'HOTA_TP', 'HOTA_FP', 'HOTA_FN']
        order = order + ['HOTA_TP(0)', 'HOTA_FP(0)', 'HOTA_FN(0)']

        order_print = [metric.replace('_', '\\_') for metric in order]

        rows = [] 
        header_row = self._row_to_latex(['Summary'] + order_print)
        rows.append(header_row)

        output = ['']
        for metric in order:
            metric_results = table_res['COMBINED_SEQ'][metric]
            summary_res = self._summary_result(metric, metric_results)
            output.append(summary_res[-1])

        rows.append(self._row_to_latex(output))

        return rows

