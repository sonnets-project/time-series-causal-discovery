import numpy as np
import stumpy

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def extract_non_overlapping_motifs(time_series, window_size, motif_quantile, circle_quantile):
    """Extract non-overlapping motif pairs and their circle from a univariate time series."""
    # Compute the matrix profile
    matrix_profile = stumpy.stump(time_series, window_size)
    profile, indices = matrix_profile[:, 0], matrix_profile[:, 1]
    quantile_threshold = np.quantile(profile, motif_quantile)

    motif_dicts, used_indices = [], set()
    
    def extract_circle(start1, start2):
        motif1, motif2 = time_series[start1:start1 + window_size], time_series[start2:start2 + window_size]
        distances = np.array([stumpy.core.mass(time_series[i:i + window_size], motif1)[0] + 
                              stumpy.core.mass(time_series[i:i + window_size], motif2)[0] 
                              for i in range(len(time_series) - window_size + 1)])
        quantile_threshold_circle = np.quantile(distances, circle_quantile)
        circle_positions, used_indices_circle = [], set()

        for idx in np.argsort(distances):
            if distances[idx] > quantile_threshold_circle:
                break
            if not any(i in used_indices_circle for i in range(idx, idx + window_size)):
                circle_positions.append(idx)
                used_indices_circle.update(range(idx, idx + window_size))

        return circle_positions

    for idx in np.argsort(profile):
        if profile[idx] > quantile_threshold:
            break
        start_left, start_right = min(idx, indices[idx]), max(idx, indices[idx])
        if not any(i in used_indices for i in range(start_left, start_left + window_size)) and not any(i in used_indices for i in range(start_right, start_right + window_size)):
            motif_dicts.append({
                "start_left": start_left,
                "start_right": start_right,
                "distance": profile[idx],
                "circle": extract_circle(start_left, start_right)
            })
            used_indices.update(range(start_left, start_left + window_size))
            used_indices.update(range(start_right, start_right + window_size))
    
    return motif_dicts


def calculate_causal_strength(motif_dicts_cause, motif_dicts_effect, max_lag):
    """Calculate the total causal strength between two lists of motif dictionaries."""
    total_causal_strength = 0.0

    for effect_dict in motif_dicts_effect:
        t_el, t_er, effect_circle = effect_dict['start_left'], effect_dict['start_right'], effect_dict['circle']
        min_distance, min_cause_idx = np.inf, -1

        for cause_idx, cause_dict in enumerate(motif_dicts_cause):
            cause_circle = cause_dict['circle']
            t_cl_candidates = [t for t in cause_circle if t < t_el]
            t_cr_candidates = [t for t in cause_circle if t < t_er]

            if t_cl_candidates and t_cr_candidates:
                t_cl, t_cr = max(t_cl_candidates), max(t_cr_candidates)
                distance = (t_el - t_cl) + (t_er - t_cr)
                if (t_el - t_cl <= max_lag) and (t_er - t_cr <= max_lag) and distance < min_distance:
                    min_distance, min_cause_idx = distance, cause_idx

        if min_distance < np.inf:
            counter = sum(1 for t_cause in motif_dicts_cause[min_cause_idx]['circle']
                          if any(0 < t_effect - t_cause <= max_lag for t_effect in effect_circle))
            partial_causal_strength = counter / len(motif_dicts_cause[min_cause_idx]['circle'])
            total_causal_strength += partial_causal_strength

    return total_causal_strength


def calculate_causal_match(motif_dicts_cause, motif_dicts_effect, max_lag, effect_used):
    num_cause_nodes = len(motif_dicts_cause)
    num_effect_nodes = len(motif_dicts_effect)
    adjacency_matrix = np.zeros((num_cause_nodes, num_effect_nodes), dtype=int)

    for effect_index, effect_motif in enumerate(motif_dicts_effect):
        if effect_used[effect_index]:
            continue
        t_el = effect_motif['start_left']
        t_er = effect_motif['start_right']

        for cause_index, cause_motif in enumerate(motif_dicts_cause):
            circle = cause_motif['circle']
            t_cl = max((t for t in circle if t < t_el), default=None)
            t_cr = max((t for t in circle if t < t_er), default=None)

            if t_cl is not None and t_cr is not None and \
               (t_el - t_cl <= max_lag) and (t_er - t_cr <= max_lag):
                adjacency_matrix[cause_index, effect_index] = 1

    csr_adj_matrix = csr_matrix(adjacency_matrix)
    matching = maximum_bipartite_matching(csr_adj_matrix, perm_type='row')
    causal_match_score = np.sum(matching != -1)
    
    return causal_match_score, matching

def would_cause_cycle(graph, start, end):
    """
    Check if adding an edge (start -> end) would create a cycle in the graph.
    """
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(neighbor for neighbor, _ in graph if neighbor == node)

    return False

def compute_graph(motif_lists, max_lag, threshold=0.05):
    num_lists = len(motif_lists)
    match_scores = np.zeros((num_lists, num_lists))
    matchings = [[None for _ in range(num_lists)] for _ in range(num_lists)]
    effect_used_lists = [[False] * len(motif_list) for motif_list in motif_lists]
    
    causal_graph = []

    while True:
        max_match_score = 0
        maxi, maxj = -1, -1
        for i in range(num_lists):
            for j in range(num_lists):
                if i != j:
                    raw_score, matching = calculate_causal_match(motif_lists[i], motif_lists[j], max_lag, effect_used_lists[j])
                    score = raw_score / len(motif_lists[i])
                    match_scores[i][j] = score
                    matchings[i][j] = matching
                    if score > max_match_score:
                        max_match_score = score
                        maxi, maxj = i, j

        if max_match_score < threshold:
            break
        
        for k in range(len(matchings[maxi][maxj])):
            if matchings[maxi][maxj][k] != -1:
                effect_used_lists[maxj][k] = True

        causal_graph.append((maxi, maxj))

    return causal_graph
